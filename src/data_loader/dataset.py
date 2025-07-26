import torch
from torch.utils.data import Dataset, random_split, DataLoader
import os
from PIL import Image
import torchvision.transforms as transforms
import logging

class ImagePosePressureDataset(Dataset):
    """
    An efficient PyTorch Dataset for loading preprocessed data.
    - Loads a single, consolidated metadata file ('ALL_DATA.pt').
    - Retrieves pose and pressure data from pre-loaded tensors instantly.
    - Loads images from file paths on-the-fly.
    - Ideal for use with DataLoader and multiple workers for high performance.
    """
    def __init__(self, metadata_file: str, transform=None):
        """
        Args:
            metadata_file (str): Path to the consolidated 'ALL_DATA.pt' file.
            transform (callable, optional): A transform to apply to the images.
                                           If None, default ToTensor and Normalize is used.
        """
        logging.info(f"Loading consolidated metadata from: {metadata_file}")
        if not os.path.exists(metadata_file):
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

        data = torch.load(metadata_file)
        self.image_paths = data['image_paths']
        self.poses = data['poses']
        self.pressures = data['pressures']

        self._len = len(self.image_paths)

        # Basic integrity check
        if not (self._len == len(self.poses) and self._len == len(self.pressures)):
            raise ValueError("Mismatched data lengths in metadata file!")

        # Define default image transformations if none are provided
        if transform is None:
            self.transform = transforms.Compose([
                # Images are already resized, so we only need to convert to tensor and normalize
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform
        
        logging.info(f"Dataset initialized successfully. Total samples: {self._len}")

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        if idx < 0 or idx >= self._len:
            raise IndexError("Index out of range")

        # Get data for the given index (pose/pressure access is instant)
        image_path = self.image_paths[idx]
        pose = self.poses[idx]
        pressure = self.pressures[idx]

        # Load the image from its path
        try:
            image = Image.open(image_path).convert('RGB')
        except FileNotFoundError:
            logging.error(f"Image file not found at {image_path}. Please check data integrity.")
            # Propagate the error. The DataLoader can be configured to handle this.
            raise
        
        # Apply transformations to the image
        image_tensor = self.transform(image)

        return {
            'image': image_tensor,
            'pose': pose,
            'pressure_map': pressure
        }

# --- Test and Usage Example ---
def main():
    """Demonstrates using the new, optimized dataset and dataloader."""
    # Path to the final, consolidated data file created by `make_single_file.py`
    metadata_path = "/scratch/avs7793/work_done/poseembroider/new_model/src/data/processed/ALL_DATA.pt"
    
    print("--- Initializing New Dataset ---")
    full_dataset = PressurePoseDataset(metadata_file=metadata_path)
    print(f"\n--- Dataset Ready. Found {len(full_dataset)} total samples. ---")

    # --- Splitting Dataset ---
    train_size = int(0.7 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    
    generator = torch.Generator().manual_seed(781)
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size], generator=generator)

    print(f"\nTraining dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    # --- Creating DataLoader (Optimized for Performance) ---
    # With this new design, using num_workers > 0 is essential for speed.
    print("\n--- Creating DataLoader (Optimized Mode) ---")
    train_loader = DataLoader(
        train_dataset, 
        batch_size=64, 
        shuffle=True, 
        num_workers=2,    # Use multiple processes to load images in the background
        pin_memory=True   # Speeds up data transfer to the GPU
    )
    
    # --- Fetch and Verify a Batch ---
    print("\nFetching first batch...")
    try:
        first_train_batch = next(iter(train_loader))
        print("\n--- Verifying Batch ---")
        print(f"Batch image shape:    {first_train_batch['image'].shape}")
        print(f"Batch pose shape:       {first_train_batch['pose'].shape}")
        print(f"Batch pressure shape:   {first_train_batch['pressure_map'].shape}")
        print("\nSuccessfully created and tested the optimized data pipeline.")
    except Exception as e:
        print(f"\nAn error occurred while fetching a batch: {e}")
        print("This could be due to issues with file paths or data corruption.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main()