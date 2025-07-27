import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import logging
import json
from datetime import datetime

import new_model.src.config as config
from new_model.src.full_model import PoseImagePressureEmbroider
from new_model.src.data_loader.dataset import ImagePosePressureDataset

# --- Configuration ---
# Set the path to the model checkpoint you want to evaluate
CHECKPOINT_PATH = "/scratch/avs7793/work_done/poseembroider/new_model/src/checkpoints/model_epoch_65.pth"
K_RECALL_VALUES = [1, 5, 10]
FEATURES_SAVE_DIR = "/scratch/avs7793/work_done/poseembroider/new_model/src/features_cache_65"

# Setup logging
def setup_logging():
    """Setup logging configuration"""
    log_dir = os.path.dirname(CHECKPOINT_PATH)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"benchmark_log_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return log_file

def x2y_recall_metrics(x_features, y_features, k_values, sstr="", batch_size=1024):
    """
    Memory-efficient calculation of recall metrics for cross-modal retrieval.
    """
    nb_x = len(x_features)
    sstrR = sstr + 'R@%d'
    recalls = {sstrR % k: 0 for k in k_values}
    
    # Process queries in batches to avoid memory issues
    for start_idx in tqdm(range(0, nb_x, batch_size), desc=f"Calculating {sstr} Recall"):
        end_idx = min(start_idx + batch_size, nb_x)
        batch_x = x_features[start_idx:end_idx]
        
        # Calculate similarity scores for this batch of queries
        batch_scores = batch_x @ y_features.t()  # Shape: (batch_size, N)
        
        # Process each query in the batch
        for i, scores in enumerate(batch_scores):
            actual_idx = start_idx + i
            # Sort scores in descending order
            _, indices_rank = scores.sort(descending=True)
            # Find the rank of the ground truth (correct pairing has same index)
            gt_rank = torch.where(indices_rank == actual_idx)[0][0].item()
            # Check if the rank is within the top K
            for k in k_values:
                if gt_rank < k:
                    recalls[sstrR % k] += 1

    # Convert to percentages
    for k in k_values:
        recalls[sstrR % k] = (recalls[sstrR % k] / nb_x) * 100.0
    
    # Log individual results immediately
    avg_recall = sum(recalls.values()) / len(recalls)
    logging.info(f"METRIC CALCULATED - {sstr}: {recalls}")
    logging.info(f"AVERAGE RECALL for {sstr}: {avg_recall:.2f}%")
    
    return recalls

def save_features(features_dict, filename):
    """Save features to disk"""
    os.makedirs(FEATURES_SAVE_DIR, exist_ok=True)
    filepath = os.path.join(FEATURES_SAVE_DIR, filename)
    torch.save(features_dict, filepath)
    logging.info(f"Features saved to: {filepath}")

def load_features(filename):
    """Load features from disk if they exist"""
    filepath = os.path.join(FEATURES_SAVE_DIR, filename)
    if os.path.exists(filepath):
        logging.info(f"Loading cached features from: {filepath}")
        return torch.load(filepath, map_location='cpu')
    return None

def infer_collection_features(model, dataloader, device):
    """
    Infers the 'ground truth' features from each individual encoder.
    """
    # Try to load cached features first
    cached_features = load_features("collection_features.pt")
    if cached_features is not None:
        return cached_features
    
    logging.info("Computing collection features (ground truth embeddings)...")
    model.eval()
    all_image_feats, all_pose_feats, all_pressure_feats = [], [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Inferring Collection Features"):
            images = batch['image'].to(device)
            poses = batch['pose'].to(device)
            pressures = batch['pressure_map'].to(device)
            
            # Get features from each individual encoder
            image_emb = model.image_encoder(images).squeeze(1).cpu()  # Move to CPU immediately
            pose_emb = model.pose_encoder(poses).squeeze(1).cpu()
            pressure_emb = model.pressure_encoder(pressures).squeeze(1).cpu()
            
            all_image_feats.append(image_emb)
            all_pose_feats.append(pose_emb)
            all_pressure_feats.append(pressure_emb)

    features_dict = {
        'image': torch.cat(all_image_feats),
        'pose': torch.cat(all_pose_feats),
        'pressure': torch.cat(all_pressure_feats)
    }
    
    # Save features for future use
    save_features(features_dict, "collection_features.pt")
    
    return features_dict

def infer_retrieval_features(model, dataloader, device):
    """
    Infers the fused-and-projected features for single and dual query types only.
    """
    # Try to load cached features first
    cached_features = load_features("retrieval_features.pt")
    if cached_features is not None:
        return cached_features
    
    logging.info("Computing retrieval features (fusion transformer outputs)...")
    model.eval()
    retrieval_features_lists = {}

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Inferring Retrieval Features"):
            images = batch['image'].to(device)
            poses = batch['pose'].to(device)
            pressures = batch['pressure_map'].to(device)
            
            # Get embeddings from individual encoders first
            # image_emb = model.image_encoder(images)    # First encoding
            # pose_emb = model.pose_encoder(poses)       # First encoding  
            # pressure_emb = model.pressure_encoder(pressures)  # First encoding
            
            # Get retrieval features from the fusion transformer
            batch_features = model.get_retrieval_features(
                images=images,      # Raw images, not embeddings
                poses=poses,        # Raw poses, not embeddings  
                pressure_maps=pressures  # Raw pressure maps, not embeddings
            )

            # Only keep single and dual query types (no full triplet)
            desired_query_types = [
                'only_image_input', 'only_pose_input', 'only_pressure_input',  # Single queries
                'image_pose_input', 'image_pressure_input', 'pose_pressure_input'  # Dual queries
            ]

            # Append results to our lists
            for query_type in desired_query_types:
                if query_type in batch_features:
                    if query_type not in retrieval_features_lists:
                        retrieval_features_lists[query_type] = {}
                    for modality, features in batch_features[query_type].items():
                        if modality not in retrieval_features_lists[query_type]:
                            retrieval_features_lists[query_type][modality] = []
                        retrieval_features_lists[query_type][modality].append(features.cpu())

    # Concatenate all batch tensors into single large tensors
    final_retrieval_features = {}
    for query_type, modality_dict in retrieval_features_lists.items():
        final_retrieval_features[query_type] = {}
        for modality, feat_list in modality_dict.items():
            final_retrieval_features[query_type][modality] = torch.cat(feat_list)
    
    # Save features for future use
    save_features(final_retrieval_features, "retrieval_features.pt")
    
    return final_retrieval_features

def benchmark_model(checkpoint_path: str):
    """
    Main function to run the benchmark evaluation.
    """
    log_file = setup_logging()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f"Using device: {device}")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    # --- Load Model ---
    model = PoseImagePressureEmbroider(latentD=config.LATENT_D).to(device)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    logging.info(f"Loaded model from {checkpoint_path}")

    # --- Load Data (using the same split as training) ---
    full_dataset = ImagePosePressureDataset(metadata_file=config.DATA_FILE)
    test_size = int(0.2 * len(full_dataset))
    train_size = len(full_dataset) - test_size
    generator = torch.Generator().manual_seed(config.SEED)
    _, test_dataset = data.random_split(full_dataset, [train_size, test_size], generator=generator)
    
    logging.info(f"Test dataset size: {len(test_dataset)}")
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False, num_workers=8, pin_memory=True)

    # --- Perform Inference ---
    logging.info("Step 1: Computing/Loading collection features...")
    collection_features = infer_collection_features(model, test_loader, device)
    torch.cuda.empty_cache()
    
    logging.info("Step 2: Computing/Loading retrieval features...")
    retrieval_features = infer_retrieval_features(model, test_loader, device)
    
    # Move features to CPU to free GPU memory for the similarity computation
    print("\nMoving features to CPU to save GPU memory...")
    for modality in collection_features:
        collection_features[modality] = collection_features[modality].cpu()
    
    for query_type in retrieval_features:
        for modality in retrieval_features[query_type]:
            retrieval_features[query_type][modality] = retrieval_features[query_type][modality].cpu()
    
    torch.cuda.empty_cache()
    
    # --- Calculate Metrics with Progressive Logging ---
    logging.info("Step 3: Calculating recall metrics...")
    all_recalls = {}
    modalities = ['image', 'pose', 'pressure']

    # 1. Single-modality queries (e.g., only_image_input -> pose)
    logging.info("=" * 50)
    logging.info("SINGLE-MODALITY QUERIES")
    logging.info("=" * 50)
    
    single_query_types = {
        'only_image_input': 'image',
        'only_pose_input': 'pose', 
        'only_pressure_input': 'pressure'
    }
    
    single_recalls = []
    for query_key, query_modality in single_query_types.items():
        if query_key in retrieval_features:
            for target_modality in modalities:
                if target_modality != query_modality:
                    logging.info(f"Computing {query_modality} → {target_modality} recall...")
                    query_feats = retrieval_features[query_key][target_modality]
                    gallery_feats = collection_features[target_modality]
                    recalls = x2y_recall_metrics(
                        query_feats, gallery_feats, K_RECALL_VALUES, 
                        sstr=f"{query_modality}→{target_modality}_"
                    )
                    all_recalls.update(recalls)
                    
                    # Add to single recalls list for summary
                    avg_recall = sum(recalls.values()) / len(recalls)
                    single_recalls.append(avg_recall)

    # 2. Dual-modality queries (e.g., image+pose -> pressure)
    logging.info("=" * 50)
    logging.info("DUAL-MODALITY QUERIES")
    logging.info("=" * 50)
    
    dual_query_types = {
        'image_pose_input': ['image', 'pose'],
        'image_pressure_input': ['image', 'pressure'],
        'pose_pressure_input': ['pose', 'pressure']
    }
    
    dual_recalls = []
    for query_key, query_modalities in dual_query_types.items():
        # logging.info(f"Computing {query_key} recall...")
        # logging.info(f"Query modalities: {query_modalities}")
        # logging.info(f"Collection modalities: {modalities}")
        # logging.info(f"Collection features: {collection_features.keys()}")
        # logging.info(f"Retrieval features: {retrieval_features.keys()}")
        if query_key in retrieval_features:
            # print(query_key)
            # print(retrieval_features.keys())
            # print("lol")
            
            # Find the target modality (the one that's missing)
            target_modality = [m for m in modalities if m not in query_modalities][0]
            
            logging.info(f"Computing {'+'.join(query_modalities)} → {target_modality} recall...")
            query_feats = retrieval_features[query_key][target_modality]
            gallery_feats = collection_features[target_modality]
            recalls = x2y_recall_metrics(
                query_feats, gallery_feats, K_RECALL_VALUES,
                sstr=f"{'+'.join(query_modalities)}→{target_modality}_"
            )
            all_recalls.update(recalls)
            
            # Add to dual recalls list for summary
            avg_recall = sum(recalls.values()) / len(recalls)
            dual_recalls.append(avg_recall)

    # --- Calculate and Log Summary Metrics ---
    logging.info("=" * 50)
    logging.info("SUMMARY METRICS")
    logging.info("=" * 50)
    
    if single_recalls:
        mean_single_recall = sum(single_recalls) / len(single_recalls)
        all_recalls['mRecall_Single'] = mean_single_recall
        logging.info(f"MEAN SINGLE-MODALITY RECALL: {mean_single_recall:.2f}%")
    
    if dual_recalls:
        mean_dual_recall = sum(dual_recalls) / len(dual_recalls)
        all_recalls['mRecall_Dual'] = mean_dual_recall
        logging.info(f"MEAN DUAL-MODALITY RECALL: {mean_dual_recall:.2f}%")
    
    # Overall mean recall
    if single_recalls and dual_recalls:
        overall_mean_recall = (mean_single_recall + mean_dual_recall) / 2
        all_recalls['mRecall_Overall'] = overall_mean_recall
        logging.info(f"OVERALL MEAN RECALL: {overall_mean_recall:.2f}%")

    # --- Save Final Results ---
    results_file = os.path.join(os.path.dirname(checkpoint_path), "benchmark_results.json")
    with open(results_file, 'w') as f:
        json.dump(all_recalls, f, indent=2)
    
    logging.info(f"All results saved to: {results_file}")
    logging.info(f"Detailed logs saved to: {log_file}")
    
    # Print final summary to console
    print(f"\n{'='*60}")
    print("FINAL BENCHMARK SUMMARY")
    print(f"{'='*60}")
    if 'mRecall_Overall' in all_recalls:
        print(f"Overall Mean Recall: {all_recalls['mRecall_Overall']:.2f}%")
    if 'mRecall_Single' in all_recalls:
        print(f"Single-Modality Mean Recall: {all_recalls['mRecall_Single']:.2f}%")
    if 'mRecall_Dual' in all_recalls:
        print(f"Dual-Modality Mean Recall: {all_recalls['mRecall_Dual']:.2f}%")

if __name__ == '__main__':
    benchmark_model(CHECKPOINT_PATH)
