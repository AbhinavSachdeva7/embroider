import torch
import torch.utils.data as data
from torch.utils.data import DataLoader, Subset
import os
import logging
import json
import numpy as np
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt

import new_model.src.config as config
from new_model.src.full_model import PoseImagePressureEmbroider
from new_model.src.data_loader.dataset import ImagePosePressureDataset

# --- Configuration ---
CHECKPOINT_PATH = "/scratch/avs7793/work_done/poseembroider/new_model/src/checkpoints/model_epoch_125.pth"
COLLECTION_FEATURES_PATH = "/scratch/avs7793/work_done/poseembroider/new_model/src/benchmark_features_cache/features_cache_125/collection_features.pt"
IMAGE_PATHS_JSON = "/scratch/avs7793/work_done/poseembroider/new_model/image_path.json"
OUTPUT_DIR = "/scratch/avs7793/work_done/poseembroider/new_model/target_results"
RESULTS_JSON = "/scratch/avs7793/work_done/poseembroider/new_model/target_results.json"

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def setup_model_dataset_and_targets():
    """Setup model, dataset, and find target indices"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    
    # Load model
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Checkpoint file not found: {CHECKPOINT_PATH}")
    
    model = PoseImagePressureEmbroider(latentD=config.LATENT_D).to(device)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    logging.info(f"Loaded model from {CHECKPOINT_PATH}")
    
    # Setup dataset with same split as benchmark.py
    full_dataset = ImagePosePressureDataset(metadata_file=config.DATA_FILE)
    test_size = int(0.2 * len(full_dataset))
    train_size = len(full_dataset) - test_size
    generator = torch.Generator().manual_seed(config.SEED)
    _, test_dataset = data.random_split(full_dataset, [train_size, test_size], generator=generator)
    
    # Load target samples and find their test indices
    with open(IMAGE_PATHS_JSON, 'r') as f:
        target_samples = json.load(f)
    
    # Create mapping from full_dataset index to test_dataset index
    full_to_test_mapping = {}
    for test_idx, full_idx in enumerate(test_dataset.indices):
        full_to_test_mapping[full_idx] = test_idx
    
    # Find test indices for our target samples
    target_test_indices = {}
    for img_path, full_idx in target_samples.items():
        if full_idx in full_to_test_mapping:
            test_idx = full_to_test_mapping[full_idx]
            target_test_indices[img_path] = {
                'full_dataset_idx': full_idx,
                'test_dataset_idx': test_idx
            }
        else:
            logging.warning(f"Full dataset index {full_idx} not found in test dataset")
    
    logging.info(f"Found {len(target_test_indices)} target samples in test dataset")
    return model, test_dataset, target_test_indices, device

def load_collection_features():
    """Load the collection features (haystack) computed by benchmark.py"""
    if not os.path.exists(COLLECTION_FEATURES_PATH):
        raise FileNotFoundError(f"Collection features not found: {COLLECTION_FEATURES_PATH}")
    
    logging.info("Loading collection features (haystack)...")
    collection_features = torch.load(COLLECTION_FEATURES_PATH, map_location='cpu')
    return collection_features

def compute_targeted_retrieval_features(model, test_dataset, target_indices_list, device):
    """Compute retrieval features for only the targeted samples"""
    logging.info(f"Computing retrieval features for {len(target_indices_list)} targeted samples...")
    
    # Create a subset dataset with only our target samples
    target_subset = Subset(test_dataset, target_indices_list)
    target_loader = DataLoader(target_subset, batch_size=min(32, len(target_indices_list)), 
                              shuffle=False, num_workers=4, pin_memory=True)
    
    retrieval_features_lists = {}
    
    with torch.no_grad():
        for batch in tqdm(target_loader, desc="Computing targeted retrieval features"):
            images = batch['image'].to(device)
            poses = batch['pose'].to(device)
            pressures = batch['pressure_map'].to(device)
            
            # Get retrieval features from fusion transformer
            batch_features = model.get_retrieval_features(
                images=images, poses=poses, pressure_maps=pressures
            )
            
            # Store only single and dual query types
            desired_query_types = [
                'only_image_input', 'only_pose_input', 'only_pressure_input',
                'image_pose_input', 'image_pressure_input', 'pose_pressure_input'
            ]
            
            for query_type in desired_query_types:
                if query_type in batch_features:
                    if query_type not in retrieval_features_lists:
                        retrieval_features_lists[query_type] = {}
                    for modality, features in batch_features[query_type].items():
                        if modality not in retrieval_features_lists[query_type]:
                            retrieval_features_lists[query_type][modality] = []
                        retrieval_features_lists[query_type][modality].append(features.cpu())
    
    # Concatenate all features
    final_retrieval_features = {}
    for query_type, modality_dict in retrieval_features_lists.items():
        final_retrieval_features[query_type] = {}
        for modality, feat_list in modality_dict.items():
            final_retrieval_features[query_type][modality] = torch.cat(feat_list)
    
    return final_retrieval_features

def get_retrieval_results_and_rank(query_features, gallery_features, query_idx, gt_test_idx, k=3):
    """Get top-k retrieval results and ground truth rank in one function"""
    query_feat = query_features[query_idx:query_idx+1]  # Shape: [1, feature_dim]
    
    # Calculate similarities with entire gallery
    similarities = query_feat @ gallery_features.t()  # Shape: [1, gallery_size]
    similarities = similarities.squeeze(0)  # Shape: [gallery_size]
    
    # Get top-k indices and scores
    top_scores, top_indices = similarities.topk(k, largest=True)
    
    # Find ground truth rank
    _, indices_rank = similarities.sort(descending=True)
    gt_rank = torch.where(indices_rank == gt_test_idx)[0][0].item()
    
    return top_indices.tolist(), top_scores.tolist(), gt_rank

# ========== VISUALIZATION FUNCTIONS (from inference2.py) ==========

def save_image(image_tensor, filepath):
    """Save denormalized image tensor (from inference2.py)"""
    if image_tensor.is_cuda:
        image_tensor = image_tensor.cpu()
        
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image_tensor = image_tensor * std + mean
    image_tensor = torch.clamp(image_tensor, 0, 1)
    
    image_np = (image_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    image_pil = Image.fromarray(image_np)
    image_pil.save(filepath)

def save_pressure_heatmap(pressure_tensor, filepath, title="Pressure Map", pressure_shape=(60, 42)):
    """Save pressure map as heatmap (from inference2.py)"""
    if pressure_tensor.is_cuda:
        pressure_tensor = pressure_tensor.cpu()
        
    pressure_map = pressure_tensor.numpy().reshape(pressure_shape)
    fig, ax = plt.subplots(figsize=(6, 8))
    im = ax.imshow(pressure_map, cmap='viridis', interpolation='nearest')
    fig.colorbar(im)
    ax.set_title(title)
    plt.savefig(filepath, bbox_inches='tight', dpi=150)
    plt.close(fig)

def create_sample_visualizations(test_dataset, sample_info, combo_results, output_dir):
    """Create visualizations for one sample's results"""
    os.makedirs(output_dir, exist_ok=True)
    
    query_idx = sample_info['test_dataset_idx']
    query_sample = test_dataset[query_idx]
    
    # Save query data
    save_image(query_sample['image'], os.path.join(output_dir, "query_image.png"))
    save_pressure_heatmap(query_sample['pressure_map'], 
                         os.path.join(output_dir, "query_pressure.png"), 
                         "Query Pressure Map")
    torch.save(query_sample['pose'], os.path.join(output_dir, "query_pose.pt"))
    
    # Save top 3 retrieved results
    top_indices = combo_results['top_3_indices']
    top_scores = combo_results.get('top_3_scores', [0.0] * len(top_indices))  # Handle case without scores
    
    for rank, (idx, score) in enumerate(zip(top_indices, top_scores), 1):
        retrieved_sample = test_dataset[idx]
        rank_dir = os.path.join(output_dir, f"rank_{rank:02d}")
        os.makedirs(rank_dir, exist_ok=True)
        
        # Save retrieved sample data
        save_image(retrieved_sample['image'], os.path.join(rank_dir, "image.png"))
        save_pressure_heatmap(retrieved_sample['pressure_map'],
                             os.path.join(rank_dir, "pressure.png"),
                             f"Rank {rank} Pressure Map")
        torch.save(retrieved_sample['pose'], os.path.join(rank_dir, "pose.pt"))
        
        # Save metadata
        metadata = {
            'rank': rank,
            'test_dataset_idx': idx,
            'full_dataset_idx': test_dataset.indices[idx],
            'similarity_score': score if isinstance(score, float) else float(score),
            'is_ground_truth': idx == query_idx
        }
        with open(os.path.join(rank_dir, "metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)
    
    # Save combo summary (matches inference2.py format)
    summary = combo_results.copy()
    summary['visualization_created'] = True
    with open(os.path.join(output_dir, "summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)

def run_needle_haystack_retrieval():
    """Main function with both JSON results and visualizations"""
    setup_logging()
    
    # Setup everything in one go
    model, test_dataset, target_test_indices, device = setup_model_dataset_and_targets()
    
    # Load collection features (haystack)
    collection_features = load_collection_features()
    total_candidates = len(collection_features['image'])
    
    # Get list of test indices for our targeted samples
    target_indices_list = [info['test_dataset_idx'] for info in target_test_indices.values()]
    
    # Compute retrieval features for ONLY our targeted samples
    targeted_retrieval_features = compute_targeted_retrieval_features(
        model, test_dataset, target_indices_list, device
    )
    
    # Define query configurations
    query_configs = [
        ('only_image_input', 'image', ['pose', 'pressure']),
        ('only_pose_input', 'pose', ['image', 'pressure']),
        ('only_pressure_input', 'pressure', ['image', 'pose']),
        ('image_pose_input', 'image+pose', ['pressure']),
        ('image_pressure_input', 'image+pressure', ['pose']),
        ('pose_pressure_input', 'pose+pressure', ['image'])
    ]
    
    # Results storage
    all_results = {}
    total_combinations = 0
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Create mapping from test_dataset_idx back to our sample info
    test_idx_to_sample_info = {}
    for img_path, info in target_test_indices.items():
        test_idx_to_sample_info[info['test_dataset_idx']] = {
            'image_path': img_path,
            'full_dataset_idx': info['full_dataset_idx']
        }
    
    # Process each targeted sample
    for target_query_idx, test_idx in enumerate(tqdm(target_indices_list, desc="Processing targeted samples")):
        sample_info = test_idx_to_sample_info[test_idx]
        img_path = sample_info['image_path']
        full_idx = sample_info['full_dataset_idx']
        
        # Create sample directory (use filename from path)
        sample_name = os.path.basename(img_path).replace('.png', '')
        sample_dir = os.path.join(OUTPUT_DIR, f"sample_{sample_name}")
        
        sample_results = {
            'image_path': img_path,
            'full_dataset_idx': full_idx,
            'test_dataset_idx': test_idx,
            'query_combinations': {}
        }
        
        # Process each query configuration
        for query_key, query_name, target_modalities in query_configs:
            if query_key not in targeted_retrieval_features:
                logging.warning(f"Query type {query_key} not found in targeted retrieval features")
                continue
            
            for target_modality in target_modalities:
                combo_name = f"{query_name}_to_{target_modality}"
                
                # Get query features and gallery features
                query_feats = targeted_retrieval_features[query_key][target_modality]
                gallery_feats = collection_features[target_modality]
                
                # Get top-3 retrieval results and ground truth rank
                top_indices, top_scores, gt_rank = get_retrieval_results_and_rank(
                    query_feats, gallery_feats, target_query_idx, test_idx, k=3
                )
                
                # Format results
                combo_results = {
                    'criteria': 'targeted_retrieval',
                    'query_type': query_name,
                    'target_modality': target_modality,
                    'query_idx': target_query_idx,
                    'test_dataset_query_idx': test_idx,
                    'ground_truth_rank': gt_rank + 1,
                    'total_candidates': total_candidates,
                    'top_3_indices': top_indices,
                    'top_3_scores': top_scores,
                    'top_3_full_dataset_indices': [test_dataset.indices[idx] for idx in top_indices]
                }
                
                sample_results['query_combinations'][combo_name] = combo_results
                
                # Create visualizations
                combo_dir = os.path.join(sample_dir, combo_name)
                create_sample_visualizations(test_dataset, 
                                           {'test_dataset_idx': test_idx}, 
                                           combo_results, combo_dir)
                
                total_combinations += 1
        
        all_results[img_path] = sample_results
        logging.info(f"Completed visualizations for {sample_name} in {sample_dir}")
    
    # Save JSON results
    logging.info(f"Saving JSON results to {RESULTS_JSON}")
    with open(RESULTS_JSON, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Print comprehensive statistics
    print_comprehensive_statistics(all_results, total_combinations, total_candidates)
    
    logging.info(f"All visualizations saved to: {OUTPUT_DIR}")

def print_comprehensive_statistics(all_results, total_combinations, total_candidates):
    """Print detailed statistics about retrieval performance"""
    print(f"\n{'='*80}")
    print("NEEDLE IN HAYSTACK RETRIEVAL RESULTS")
    print(f"{'='*80}")
    
    # Collect all ground truth ranks
    gt_ranks = []
    combo_stats = {}
    
    for sample_results in all_results.values():
        for combo_name, combo_results in sample_results['query_combinations'].items():
            gt_rank = combo_results['ground_truth_rank']
            gt_ranks.append(gt_rank)
            
            if combo_name not in combo_stats:
                combo_stats[combo_name] = []
            combo_stats[combo_name].append(gt_rank)
    
    # Overall statistics
    print(f"Total samples processed: {len(all_results)}")
    print(f"Total query combinations: {total_combinations}")
    print(f"Gallery size (total candidates): {total_candidates}")
    print(f"Expected combinations: {len(all_results) * 9}")
    
    if gt_ranks:
        print(f"\nOVERALL RETRIEVAL PERFORMANCE:")
        print(f"  Mean rank: {np.mean(gt_ranks):.2f}")
        print(f"  Median rank: {np.median(gt_ranks):.2f}")
        print(f"  Min rank: {min(gt_ranks)}")
        print(f"  Max rank: {max(gt_ranks)}")
        
        # Success rates
        perfect_matches = sum(1 for r in gt_ranks if r == 1)
        top3_matches = sum(1 for r in gt_ranks if r <= 3)
        top10_matches = sum(1 for r in gt_ranks if r <= 10)
        
        print(f"\nSUCCESS RATES:")
        print(f"  Rank 1 (Perfect): {perfect_matches}/{len(gt_ranks)} ({perfect_matches/len(gt_ranks)*100:.1f}%)")
        print(f"  Top 3: {top3_matches}/{len(gt_ranks)} ({top3_matches/len(gt_ranks)*100:.1f}%)")
        print(f"  Top 10: {top10_matches}/{len(gt_ranks)} ({top10_matches/len(gt_ranks)*100:.1f}%)")

if __name__ == "__main__":
    run_needle_haystack_retrieval()