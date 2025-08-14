import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import logging
import json
import random
import numpy as np
from datetime import datetime
from PIL import Image
import matplotlib.pyplot as plt

import new_model.src.config as config
from new_model.src.full_model import PoseImagePressureEmbroider
from new_model.src.data_loader.dataset import ImagePosePressureDataset

# --- Configuration ---
CHECKPOINT_PATH = "/scratch/avs7793/work_done/poseembroider/new_model/src/checkpoints/model_epoch_125.pth"
OUTPUT_DIR = "/scratch/avs7793/work_done/poseembroider/visualization_results"
FEATURES_SAVE_DIR = "/scratch/avs7793/work_done/poseembroider/new_model/src/benchmark_features_cache/features_cache_125"
BATCH_SIZE = 100  # Process samples in batches of this size

class RetrievalVisualizer:
    """
    Finds and visualizes specific retrieval performance examples for all
    possible query combinations.
    """
    CRITERIA_TEMPLATE = {
        'best_rank_1': {'found': False, 'condition': lambda rank: rank == 0},
        'good_rank_2_5': {'found': False, 'condition': lambda rank: 1 <= rank <= 4},
        'moderate_rank_6_10': {'found': False, 'condition': lambda rank: 5 <= rank <= 9},
        'poor_rank_gt_10': {'found': False, 'condition': lambda rank: rank >= 10},
        'random': {'found': False, 'condition': lambda rank: True}
    }
    
    def __init__(self, checkpoint_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.checkpoint_path = checkpoint_path
        
        self.found_examples = {}  # Nested dict: {combo_str: {criteria: example}}
        self.sample_cache = {}    # Cache for storing individual samples when needed
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        
    def setup_model_and_data(self):
        """Load model and prepare dataset like in benchmark.py"""
        logging.info(f"Using device: {self.device}")
        
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {self.checkpoint_path}")
        
        # Load Model
        self.model = PoseImagePressureEmbroider(latentD=config.LATENT_D).to(self.device)
        checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        logging.info(f"Loaded model from {self.checkpoint_path}")
        
        # Load Data (using same split as training)
        full_dataset = ImagePosePressureDataset(metadata_file=config.DATA_FILE)
        test_size = int(0.2 * len(full_dataset))
        train_size = len(full_dataset) - test_size
        generator = torch.Generator().manual_seed(config.SEED)
        _, self.test_dataset = data.random_split(full_dataset, [train_size, test_size], generator=generator)
        
        logging.info(f"Test dataset size: {len(self.test_dataset)}")
        
    def load_or_compute_features(self):
        """Load cached features or compute them"""
        # Try to load cached features
        collection_path = os.path.join(FEATURES_SAVE_DIR, "collection_features.pt")
        retrieval_path = os.path.join(FEATURES_SAVE_DIR, "retrieval_features.pt")
        
        if os.path.exists(collection_path) and os.path.exists(retrieval_path):
            logging.info("Loading cached features...")
            self.collection_features = torch.load(collection_path, map_location='cpu')
            self.retrieval_features = torch.load(retrieval_path, map_location='cpu')
            return
            
        logging.info("Computing features (this may take a while)...")
        test_loader = DataLoader(self.test_dataset, batch_size=32, shuffle=False, num_workers=4)
        
        # Compute collection features (ground truth embeddings)
        self.collection_features = self._infer_collection_features(test_loader)
        
        # Compute retrieval features (fusion transformer outputs) 
        self.retrieval_features = self._infer_retrieval_features(test_loader)
        
    def _infer_collection_features(self, dataloader):
        """Compute ground truth features from individual encoders"""
        logging.info("Computing collection features...")
        all_image_feats, all_pose_feats, all_pressure_feats = [], [], []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Collection Features"):
                images = batch['image'].to(self.device)
                poses = batch['pose'].to(self.device) 
                pressures = batch['pressure_map'].to(self.device)
                
                # Get features from individual encoders
                image_emb = self.model.image_encoder(images).squeeze(1).cpu()
                pose_emb = self.model.pose_encoder(poses).squeeze(1).cpu()
                pressure_emb = self.model.pressure_encoder(pressures).squeeze(1).cpu()
                
                all_image_feats.append(image_emb)
                all_pose_feats.append(pose_emb)
                all_pressure_feats.append(pressure_emb)
        
        return {
            'image': torch.cat(all_image_feats),
            'pose': torch.cat(all_pose_feats), 
            'pressure': torch.cat(all_pressure_feats)
        }
    
    def _infer_retrieval_features(self, dataloader):
        """Compute retrieval features from fusion transformer"""
        logging.info("Computing retrieval features...")
        retrieval_features_lists = {}
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Retrieval Features"):
                images = batch['image'].to(self.device)
                poses = batch['pose'].to(self.device)
                pressures = batch['pressure_map'].to(self.device)
                
                # Get retrieval features
                batch_features = self.model.get_retrieval_features(
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
        
    def find_ranking_examples(self):
        """Find examples for each criterion for every query combination."""
        logging.info("Searching for ranking examples for all combinations...")

        query_configs = [
            ('only_image_input', 'image', ['pose', 'pressure']),
            ('only_pose_input', 'pose', ['image', 'pressure']),
            ('only_pressure_input', 'pressure', ['image', 'pose']),
            ('image_pose_input', 'image+pose', ['pressure']),
            ('image_pressure_input', 'image+pressure', ['pose']),
            ('pose_pressure_input', 'pose+pressure', ['image'])
        ]

        for query_key, query_name, target_modalities in query_configs:
            if query_key not in self.retrieval_features:
                continue

            for target_modality in target_modalities:
                combo_str = f"{query_name}_to_{target_modality}"
                logging.info(f"--- Processing combination: {combo_str} ---")

                # Initialize storage for this combination
                self.found_examples[combo_str] = {}
                criteria_for_combo = {k: v.copy() for k, v in self.CRITERIA_TEMPLATE.items()}
                
                random_candidates_for_combo = []

                query_feats = self.retrieval_features[query_key][target_modality]
                gallery_feats = self.collection_features[target_modality]

                total_queries = len(query_feats)
                for start_idx in tqdm(range(0, total_queries, BATCH_SIZE), desc=f"Scanning {combo_str}"):
                    if all(info['found'] for name, info in criteria_for_combo.items() if name != 'random'):
                        break

                    end_idx = min(start_idx + BATCH_SIZE, total_queries)
                    batch_feats = query_feats[start_idx:end_idx]
                    
                    similarities = batch_feats @ gallery_feats.t()

                    for batch_query_idx, query_idx in enumerate(range(start_idx, end_idx)):
                        if all(info['found'] for name, info in criteria_for_combo.items() if name != 'random'):
                            break

                        scores = similarities[batch_query_idx]
                        _, indices_rank = scores.sort(descending=True)
                        gt_rank = torch.where(indices_rank == query_idx)[0][0].item()

                        example_data = {
                            'query_key': query_key,
                            'query_name': query_name,
                            'target_modality': target_modality,
                            'query_idx': query_idx,
                            'gt_rank': gt_rank,
                            'scores': scores,
                            'indices_rank': indices_rank
                        }
                        
                        random_candidates_for_combo.append(example_data)

                        for criteria_name, criteria_info in criteria_for_combo.items():
                            if criteria_name == 'random': continue

                            if not criteria_info['found'] and criteria_info['condition'](gt_rank):
                                self.found_examples[combo_str][criteria_name] = example_data
                                criteria_info['found'] = True
                                logging.info(f"Found {criteria_name} for {combo_str} (GT rank: {gt_rank + 1})")

                # After checking all queries for this combo, select a random one
                if random_candidates_for_combo and 'random' not in self.found_examples[combo_str]:
                    random_example = random.choice(random_candidates_for_combo)
                    self.found_examples[combo_str]['random'] = random_example
                    logging.info(f"Selected random example for {combo_str} (GT rank: {random_example['gt_rank'] + 1})")

                # Log any missing criteria for this combo
                missing = [name for name in self.CRITERIA_TEMPLATE if name not in self.found_examples[combo_str]]
                if missing:
                    logging.warning(f"Could not find examples for {combo_str} for criteria: {missing}")

    def get_sample(self, idx):
        """Get a sample from the dataset, using cache if available"""
        if idx not in self.sample_cache:
            self.sample_cache[idx] = self.test_dataset[idx]
        return self.sample_cache[idx]
            
    def save_image(self, image_tensor, filepath):
        """Save denormalized image tensor"""
        if image_tensor.is_cuda:
            image_tensor = image_tensor.cpu()
            
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image_tensor = image_tensor * std + mean
        image_tensor = torch.clamp(image_tensor, 0, 1)
        
        image_np = (image_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        image_pil = Image.fromarray(image_np)
        image_pil.save(filepath)
        
    def save_pressure_heatmap(self, pressure_tensor, filepath, title="Pressure Map", pressure_shape=(60, 42)):
        """Save pressure map as heatmap"""
        if pressure_tensor.is_cuda:
            pressure_tensor = pressure_tensor.cpu()
            
        pressure_map = pressure_tensor.numpy().reshape(pressure_shape)
        fig, ax = plt.subplots(figsize=(6, 8))
        im = ax.imshow(pressure_map, cmap='viridis', interpolation='nearest')
        fig.colorbar(im)
        ax.set_title(title)
        plt.savefig(filepath, bbox_inches='tight', dpi=150)
        plt.close(fig)
        
    def create_visualizations(self):
        """Create comprehensive visualizations for all found examples and combinations."""
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        for combo_str, combo_examples in self.found_examples.items():
            combo_dir = os.path.join(OUTPUT_DIR, combo_str)
            os.makedirs(combo_dir, exist_ok=True)
            logging.info(f"Creating visualizations for {combo_str} in {combo_dir}")

            for criteria_name, example in combo_examples.items():
                example_dir = os.path.join(combo_dir, criteria_name)
                os.makedirs(example_dir, exist_ok=True)
                
                query_idx = example['query_idx']
                gt_rank = example['gt_rank']
                scores = example['scores']
                indices_rank = example['indices_rank']
                
                # Get query sample (using cache)
                query_sample = self.get_sample(query_idx)
                
                # Save query data
                self.save_image(query_sample['image'], os.path.join(example_dir, "query_image.png"))
                self.save_pressure_heatmap(query_sample['pressure_map'], 
                                         os.path.join(example_dir, "query_pressure.png"), 
                                         "Query Pressure Map")
                torch.save(query_sample['pose'], os.path.join(example_dir, "query_pose.pt"))
                
                # Save top retrieved results
                top_k = min(10, len(indices_rank))
                retrieved_indices = indices_rank[:top_k].tolist()
                retrieved_scores = scores[indices_rank[:top_k]].tolist()
                
                for rank, (idx, score) in enumerate(zip(retrieved_indices, retrieved_scores)):
                    retrieved_sample = self.get_sample(idx)
                    rank_dir = os.path.join(example_dir, f"rank_{rank+1:02d}")
                    os.makedirs(rank_dir, exist_ok=True)
                    
                    # Save retrieved sample data
                    self.save_image(retrieved_sample['image'], 
                                  os.path.join(rank_dir, "image.png"))
                    self.save_pressure_heatmap(retrieved_sample['pressure_map'],
                                             os.path.join(rank_dir, "pressure.png"),
                                             f"Rank {rank+1} Pressure Map")
                    torch.save(retrieved_sample['pose'], 
                              os.path.join(rank_dir, "pose.pt"))
                    
                    # Save metadata
                    metadata = {
                        'rank': rank + 1,
                        'sample_idx': idx,
                        'similarity_score': score,
                        'is_ground_truth': idx == query_idx
                    }
                    with open(os.path.join(rank_dir, "metadata.json"), 'w') as f:
                        json.dump(metadata, f, indent=2)
                
                # Save example summary
                summary = {
                    'criteria': criteria_name,
                    'query_type': example['query_name'],
                    'target_modality': example['target_modality'],
                    'query_idx': query_idx,
                    'ground_truth_rank': gt_rank + 1,
                    'total_candidates': len(scores),
                    'top_10_indices': retrieved_indices,
                    'top_10_scores': retrieved_scores
                }
                
                with open(os.path.join(example_dir, "summary.json"), 'w') as f:
                    json.dump(summary, f, indent=2)
                
                logging.info(f"Saved visualization for {criteria_name} in {example_dir}")
            
    def run(self):
        """Main execution pipeline"""
        logging.info("Starting retrieval visualization...")
        
        # Setup
        self.setup_model_and_data()
        
        # Load or compute features
        self.load_or_compute_features()
        
        # Find examples
        self.find_ranking_examples()
        
        # Create visualizations
        self.create_visualizations()
        
        logging.info(f"Visualization complete! Results saved to: {OUTPUT_DIR}")
        
        # Print summary
        print("\n" + "="*60)
        print("VISUALIZATION SUMMARY")
        print("="*60)
        for combo_str, combo_examples in self.found_examples.items():
            print(f"\n--- Combination: {combo_str} ---")
            if not combo_examples:
                print("  No examples found for this combination.")
                continue
            for criteria_name, example in sorted(combo_examples.items()):
                print(f"  {criteria_name:<20}: GT rank: {example['gt_rank'] + 1}, Query Index: {example['query_idx']}")

if __name__ == "__main__":
    visualizer = RetrievalVisualizer(CHECKPOINT_PATH)
    visualizer.run()