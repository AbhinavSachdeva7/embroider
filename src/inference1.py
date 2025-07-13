import torch
import numpy as np
import os
import argparse
from PIL import Image
import json
from tqdm import tqdm
import matplotlib.pyplot as plt

from new_model.src.full_model import PoseImagePressureEmbroider
from new_model.src.data_loader.dataset import PressurePoseImageDataset
import new_model.src.config as config

class ModelInference:
    """
    Inference class for PoseImagePressureEmbroider model.
    Tests retrieval capabilities across different modality combinations.
    """
    
    def __init__(self, model_checkpoint_path, test_data_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.test_data_path = test_data_path
        
        # Load model
        print(f"Loading model from: {model_checkpoint_path}")
        self.model = PoseImagePressureEmbroider(latentD=config.LATENT_D).to(self.device)
        self.model.load_state_dict(torch.load(model_checkpoint_path, map_location=self.device))
        self.model.eval()
        print(f"Model loaded on {self.device}")
        
        # Load test dataset
        print(f"Loading test dataset from: {test_data_path}")
        self.dataset = PressurePoseImageDataset(test_data_path, seed=781)
        print(f"Test dataset size: {len(self.dataset)}")
        
    def extract_features(self, batch_size=32, max_samples=1000):
        """
        Extract features for all samples in the dataset.
        Returns dictionaries of features for each modality.
        """
        print("Extracting features from test dataset...")
        
        # Limit samples for faster inference
        num_samples = min(len(self.dataset), max_samples)
        
        # Storage for features
        image_features = []
        pose_features = []
        pressure_features = []
        
        with torch.no_grad():
            for i in tqdm(range(0, num_samples, batch_size), desc="Extracting features"):
                batch_end = min(i + batch_size, num_samples)
                batch_samples = [self.dataset[j] for j in range(i, batch_end)]
                
                # Prepare batch
                images = torch.stack([sample['image'] for sample in batch_samples]).to(self.device)
                poses = torch.stack([sample['pose'] for sample in batch_samples]).to(self.device)
                pressure_maps = torch.stack([sample['pressure_map'] for sample in batch_samples]).to(self.device)
                
                # Get embeddings from each encoder
                image_emb = self.model.image_encoder(images)
                pose_emb = self.model.pose_encoder(poses)
                pressure_emb = self.model.pressure_encoder(pressure_maps)
                
                # Flatten to (batch, 512) and store
                image_features.append(image_emb.squeeze(1).cpu())
                pose_features.append(pose_emb.squeeze(1).cpu())
                pressure_features.append(pressure_emb.squeeze(1).cpu())
        
        all_image_features = torch.cat(image_features, dim=0)
        all_pose_features = torch.cat(pose_features, dim=0)
        all_pressure_features = torch.cat(pressure_features, dim=0)
        
        print(f"Extracted features from {all_image_features.shape[0]} samples")
        
        return {
            'image': all_image_features,
            'pose': all_pose_features, 
            'pressure': all_pressure_features
        }
    
    def test_retrieval(self, features, query_idx=0, top_k=5):
        """
        Test retrieval using different modality combinations.
        """
        print(f"\n=== Testing Retrieval for Query (Shuffled Index): {query_idx} ===")
        
        results = {}
        features_on_device = {k: v.to(self.device) for k, v in features.items()}
        
        query_sample = self.dataset[query_idx]
        query_images = query_sample['image'].unsqueeze(0).to(self.device)
        query_poses = query_sample['pose'].unsqueeze(0).to(self.device)
        query_pressure = query_sample['pressure_map'].unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Get all possible projected features from our query sample
            # This call will handle single, dual, and triplet fusions.
            retrieval_features = self.model.get_retrieval_features(
                images=query_images,
                poses=query_poses,
                pressure_maps=query_pressure
            )

            # Now, for each fusion type, retrieve from each target modality
            for fusion_key, projected_features in retrieval_features.items():
                print(f"\n--- Testing Query from Fused '{fusion_key}' ---")
                
                # 'projected_features' is a dict: {'image': tensor, 'pose': tensor, 'pressure': tensor}
                # These are the query features projected into each space.
                
                retrieval_results_for_fusion = {}
                for target_modality, query_feature in projected_features.items():
                    # 'target_modality' is 'image', 'pose', or 'pressure'
                    # 'query_feature' is the fused embedding projected into that space
                    
                    target_library = features_on_device[target_modality] # The library to search
                    
                    # Compute similarities
                    similarities = torch.mm(query_feature, target_library.t()).squeeze(0)
                    
                    # Get top-k indices
                    _, top_indices = similarities.topk(top_k)
                    
                    retrieval_results_for_fusion[f"retrieve_{target_modality}"] = top_indices.cpu().tolist()
                
                results[fusion_key] = retrieval_results_for_fusion
        
        return results

    def _save_single_visualization(self, sample, directory, prefix):
        """Saves image, pose, and pressure for a single sample."""
        self.save_pil_image(sample['image'], os.path.join(directory, f"{prefix}_image.png"))
        torch.save(sample['pose'], os.path.join(directory, f"{prefix}_pose.pt"))
        self.save_pressure_heatmap(sample['pressure_map'], os.path.join(directory, f"{prefix}_pressure_heatmap.png"))

    def save_pil_image(self, image_tensor, filepath):
        """Helper to save a single denormalized image tensor."""
        if image_tensor.is_cuda: image_tensor = image_tensor.cpu()
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image_tensor = image_tensor * std + mean
        image_tensor = torch.clamp(image_tensor, 0, 1)
        image_pil = Image.fromarray((image_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
        image_pil.save(filepath)

    def save_pressure_heatmap(self, pressure_tensor, filepath, pressure_shape=(60, 42)):
        """Saves a pressure map tensor as a heatmap image."""
        if pressure_tensor.is_cuda: pressure_tensor = pressure_tensor.cpu()
        pressure_map = pressure_tensor.numpy().reshape(pressure_shape)
        fig, ax = plt.subplots()
        im = ax.imshow(pressure_map, cmap='viridis', interpolation='nearest')
        fig.colorbar(im)
        ax.set_title("Pressure Map")
        plt.savefig(filepath, bbox_inches='tight')
        plt.close(fig)

    def save_retrieval_visualizations(self, results, output_dir, query_idx=0):
        """Saves detailed visualizations for the query and retrieved samples."""
        visualizations_dir = os.path.join(output_dir, "visualizations_1k")
        os.makedirs(visualizations_dir, exist_ok=True)
        print(f"Saving detailed visualizations to: {visualizations_dir}")

        original_query_idx = self.dataset.indices[query_idx]
        query_dir = os.path.join(visualizations_dir, f"query_shuffled_{query_idx}_original_{original_query_idx}")
        os.makedirs(query_dir, exist_ok=True)
        self._save_single_visualization(self.dataset[query_idx], query_dir, "query")
        with open(os.path.join(query_dir, "info.txt"), 'w') as f:
            f.write(f"--- Query ---\nShuffled Idx: {query_idx}\nOriginal Idx: {original_query_idx}")

        for test_name, test_results in results.items():
            if "error" in test_results or not test_results: continue
            
            test_dir = os.path.join(visualizations_dir, test_name)
            os.makedirs(test_dir, exist_ok=True)
            
            if "retrieve_image" in test_results:
                for rank, retrieved_shuffled_idx in enumerate(test_results["retrieve_image"]):
                    original_retrieved_idx = self.dataset.indices[retrieved_shuffled_idx]
                    result_dir = os.path.join(test_dir, f"rank_{rank+1}_shuffled_{retrieved_shuffled_idx}_original_{original_retrieved_idx}")
                    os.makedirs(result_dir, exist_ok=True)
                    self._save_single_visualization(self.dataset[retrieved_shuffled_idx], result_dir, "retrieved")
                    with open(os.path.join(result_dir, "info.txt"), 'w') as f:
                        f.write(f"--- Retrieved (Rank {rank+1}) ---\nQuery: '{test_name}'\nShuffled Idx: {retrieved_shuffled_idx}\nOriginal Idx: {original_retrieved_idx}")

    def save_results(self, results, output_dir="inference_results_18k", query_idx=0):
        """Save inference results to files."""
        os.makedirs(output_dir, exist_ok=True)
        results_file = os.path.join(output_dir, "retrieval_results.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {results_file}")
        self.save_retrieval_visualizations(results, output_dir, query_idx)

def main():
    parser = argparse.ArgumentParser(description='Inference for PoseImagePressureEmbroider')
    parser.add_argument('--checkpoint', type=str, required=False, 
                        help='Path to model checkpoint', default='/scratch/avs7793/work_done/poseembroider/new_model/src/checkpoints/model_epoch_5.pth')
    parser.add_argument('--test_data', type=str, required=False,
                        help='Path to test dataset', default='/scratch/avs7793/work_done/poseembroider/new_model/src/data/combined_subject1_take2.pt')
    parser.add_argument('--output_dir', type=str, default='/scratch/avs7793/work_done/poseembroider/new_model/inference_results',
                        help='Directory to save results')
    parser.add_argument('--query_idx', type=int, default=0,
                        help='Index of query sample (from shuffled list)')
    parser.add_argument('--top_k', type=int, default=5,
                        help='Number of top retrievals')
    parser.add_argument('--max_samples', type=int, default=1000,
                        help='Maximum samples to build feature library from')
    
    args = parser.parse_args()
    
    inference = ModelInference(args.checkpoint, args.test_data)
    features = inference.extract_features(max_samples=args.max_samples)
    results = inference.test_retrieval(features, query_idx=args.query_idx, top_k=args.top_k)
    inference.save_results(results, args.output_dir, query_idx=args.query_idx)
    
    print(f"\nInference completed! Results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()