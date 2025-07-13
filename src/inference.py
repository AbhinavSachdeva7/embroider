import torch
import numpy as np
import os
import argparse
from PIL import Image
import json
from tqdm import tqdm

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
        self.dataset = PressurePoseImageDataset(test_data_path, seed=42)
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
                image_emb = self.model.image_encoder(images)  # Shape: (batch, 1, 512)
                pose_emb = self.model.pose_encoder(poses)     # Shape: (batch, 1, 512)
                pressure_emb = self.model.pressure_encoder(pressure_maps)  # Shape: (batch, 1, 512)
                
                # Flatten to (batch, 512) and store
                image_features.append(image_emb.squeeze(1).cpu())
                pose_features.append(pose_emb.squeeze(1).cpu())
                pressure_features.append(pressure_emb.squeeze(1).cpu())
        
        # Concatenate all features
        all_image_features = torch.cat(image_features, dim=0)  # Shape: (num_samples, 512)
        all_pose_features = torch.cat(pose_features, dim=0)
        all_pressure_features = torch.cat(pressure_features, dim=0)
        
        print(f"Extracted features: {all_image_features.shape[0]} samples")
        
        return {
            'image': all_image_features,
            'pose': all_pose_features, 
            'pressure': all_pressure_features
        }
    
    def test_retrieval(self, features, query_idx=0, top_k=5):
        """
        Test retrieval using different modality combinations.
        
        Args:
            features: Dictionary with extracted features
            query_idx: Index of query sample
            top_k: Number of top results to retrieve
        """
        print(f"\n=== Testing Retrieval for Query Index: {query_idx} ===")
        
        results = {}
        
        # Get query sample
        query_sample = self.dataset[query_idx]
        query_images = query_sample['image'].unsqueeze(0).to(self.device)
        query_poses = query_sample['pose'].unsqueeze(0).to(self.device)
        query_pressure = query_sample['pressure_map'].unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Test different query modality combinations
            test_cases = [
                ("Image Only", {"images": query_images, "poses": None, "pressure_maps": None}),
                ("Pose Only", {"images": None, "poses": query_poses, "pressure_maps": None}),
                ("Pressure Only", {"images": None, "poses": None, "pressure_maps": query_pressure}),
                ("Image + Pose", {"images": query_images, "poses": query_poses, "pressure_maps": None}),
                ("Image + Pressure", {"images": query_images, "poses": None, "pressure_maps": query_pressure}),
                ("Pose + Pressure", {"images": None, "poses": query_poses, "pressure_maps": query_pressure}),
                ("All Modalities", {"images": query_images, "poses": query_poses, "pressure_maps": query_pressure})
            ]
            
            for test_name, inputs in test_cases:
                print(f"\n--- Testing: {test_name} ---")
                
                try:
                    # Get fused embeddings from model
                    loss_dict = self.model(**inputs, single_partials=True, dual_partials=True, triplet_partial=True)
                    
                    # The model returns loss dict, but we need the embeddings
                    # Let's get the raw embeddings instead
                    available_modalities = []
                    if inputs["images"] is not None:
                        available_modalities.append("image")
                    if inputs["poses"] is not None:
                        available_modalities.append("pose") 
                    if inputs["pressure_maps"] is not None:
                        available_modalities.append("pressure")
                    
                    # For now, let's test single modality retrieval
                    if len(available_modalities) == 1:
                        query_modality = available_modalities[0]
                        query_feature = None
                        
                        if query_modality == "image":
                            query_feature = self.model.image_encoder(inputs["images"]).squeeze(1)
                        elif query_modality == "pose":
                            query_feature = self.model.pose_encoder(inputs["poses"]).squeeze(1)
                        elif query_modality == "pressure":
                            query_feature = self.model.pressure_encoder(inputs["pressure_maps"]).squeeze(1)
                        
                        # Test retrieval for each target modality
                        retrieval_results = {}
                        for target_modality in ['image', 'pose', 'pressure']:
                            target_features = features[target_modality]
                            
                            # Move all features to the same device as the model
                            features_on_device = {
                                k: v.to(self.device) for k, v in features.items()
                            }
                            
                            # Compute similarities
                            similarities = torch.mm(query_feature, features_on_device[target_modality].t()).squeeze(0)
                            
                            # Get top-k indices
                            _, top_indices = similarities.topk(top_k)
                            retrieval_results[f"retrieve_{target_modality}"] = top_indices.tolist()
                            
                            print(f"  Top {top_k} {target_modality} similarities: {similarities[top_indices].tolist()}")
                        
                        results[test_name] = retrieval_results
                        
                except Exception as e:
                    print(f"  Error in {test_name}: {e}")
                    results[test_name] = {"error": str(e)}
        
        return results
    
    def save_retrieved_images(self, results, features, output_dir, query_idx=0):
        """Save images that were actually retrieved by the model."""
        retrieved_images_dir = os.path.join(output_dir, "retrieved_images")
        os.makedirs(retrieved_images_dir, exist_ok=True)
        
        # Get the original query sample
        query_sample = self.dataset[query_idx]
        
        # Save query image
        self.save_single_image(query_sample['image'], 
                              os.path.join(retrieved_images_dir, f"query_{query_idx}.png"))
        
        # For each test case, save the retrieved images
        for test_name, test_results in results.items():
            if "error" in test_results:
                continue
            
            test_dir = os.path.join(retrieved_images_dir, test_name.replace(" ", "_").lower())
            os.makedirs(test_dir, exist_ok=True)
            
            # If this test case retrieved images
            if "retrieve_image" in test_results:
                retrieved_indices = test_results["retrieve_image"]
                
                for rank, idx in enumerate(retrieved_indices):
                    # Get the actual retrieved sample
                    retrieved_sample = self.dataset[idx]
                    retrieved_image = retrieved_sample['image']
                    
                    # Save retrieved image
                    filename = f"rank_{rank+1}_idx_{idx}.png"
                    self.save_single_image(retrieved_image, 
                                         os.path.join(test_dir, filename))
                    
                    # Also save some info about this retrieved sample
                    info_filename = f"rank_{rank+1}_idx_{idx}_info.txt"
                    info_path = os.path.join(test_dir, info_filename)
                    with open(info_path, 'w') as f:
                        f.write(f"Retrieved Index: {idx}\n")
                        f.write(f"Rank: {rank+1}\n")
                        f.write(f"Query Test: {test_name}\n")
                        f.write(f"Image Shape: {retrieved_image.shape}\n")
                        f.write(f"Pose Shape: {retrieved_sample['pose'].shape}\n")
                        f.write(f"Pressure Shape: {retrieved_sample['pressure_map'].shape}\n")

    def save_single_image(self, image_tensor, filepath):
        """Helper to save a single image tensor."""
        # Denormalize
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image_tensor = image_tensor * std + mean
        image_tensor = torch.clamp(image_tensor, 0, 1)
        
        # Convert to PIL and save
        image_np = (image_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        image_pil = Image.fromarray(image_np)
        image_pil.save(filepath)

    def save_results(self, results, output_dir="inference_results", query_idx=0, features=None):
        """Save inference results to files."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save retrieval results as JSON
        results_file = os.path.join(output_dir, "retrieval_results.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {results_file}")
        
        # Save retrieved images instead of sample images
        if features is not None:
            self.save_retrieved_images(results, features, output_dir, query_idx)
            print(f"Retrieved images saved to: {output_dir}/retrieved_images/")

def main():
    parser = argparse.ArgumentParser(description='Inference for PoseImagePressureEmbroider')
    parser.add_argument('--checkpoint', type=str, required=False, 
                        help='Path to model checkpoint (.pth file)', default='/scratch/avs7793/work_done/poseembroider/new_model/src/checkpoints/model_epoch_5.pth')
    parser.add_argument('--test_data', type=str, required=False,
                        help='Path to test dataset (.pt file)', default='/scratch/avs7793/work_done/poseembroider/new_model/src/data/combined_subject1_take1.pt')
    parser.add_argument('--output_dir', type=str, default='/scratch/avs7793/work_done/poseembroider/new_model/inference_results',
                        help='Directory to save results')
    parser.add_argument('--query_idx', type=int, default=0,
                        help='Index of query sample to test')
    parser.add_argument('--top_k', type=int, default=5,
                        help='Number of top retrievals')
    parser.add_argument('--max_samples', type=int, default=18000,
                        help='Maximum samples to use for feature extraction')
    
    args = parser.parse_args()
    
    # Initialize inference
    inference = ModelInference(args.checkpoint, args.test_data)
    
    # Extract features
    features = inference.extract_features(max_samples=args.max_samples)
    
    # Test retrieval
    results = inference.test_retrieval(features, query_idx=args.query_idx, top_k=args.top_k)
    
    # Save results (now includes retrieved images)
    inference.save_results(results, args.output_dir, query_idx=args.query_idx, features=features)
    
    print(f"\nInference completed! Results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()