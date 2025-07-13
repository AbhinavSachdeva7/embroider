import torch
from torch.utils.data import Dataset, random_split, DataLoader
import os
import glob
from tqdm import tqdm
import json
import re
import gc
import psutil
# import new_model.src.config as config

class MultiFilePressurePoseDataset(Dataset):
    """
    A memory-safe PyTorch Dataset for loading from multiple .pt files.
    - Uses a JSON cache for instant metadata loading after the first run.
    - Builds the cache by scanning small source files for maximum speed.
    - NO in-memory caching to avoid memory accumulation issues.
    """
    
    def __init__(self, data_directory: str, source_data_root: str):
        """
        Args:
            data_directory (str): Path to the directory of 'combined_*.pt' files.
            source_data_root (str): Path to the root of the original data (e.g., '.../data/').
        """
        print("Initializing MultiFile Dataset...")
        self.data_directory = data_directory
        self.source_data_root = source_data_root
        metadata_cache_path = os.path.join(self.data_directory, "_metadata_cache.json")

        if os.path.exists(metadata_cache_path):
            print(f"Loading metadata from cache: {metadata_cache_path}")
            with open(metadata_cache_path, 'r') as f:
                metadata = json.load(f)
            self.file_map = metadata['file_map']
            self.cumulative_samples = metadata['cumulative_samples']
            self._len = self.cumulative_samples[-1]
        else:
            print("Metadata cache not found. Performing one-time, ultra-fast scan...")
            
            combined_files = sorted(glob.glob(os.path.join(self.data_directory, "combined_*.pt")))
            if not combined_files:
                raise FileNotFoundError(f"No 'combined_*.pt' files found in directory: {self.data_directory}")

            self.file_map = []
            self.cumulative_samples = [0]
            total_samples = 0
            
            for combined_path in tqdm(combined_files, desc="Fast-scanning source files"):
                # Parse subject and take from the combined filename
                match = re.search(r"combined_subject(\d+)_take(\d+)\.pt", os.path.basename(combined_path))
                if not match:
                    continue
                
                subject, take = match.groups()
                
                # Construct path to the small, corresponding pose file
                pose_path = os.path.join(self.source_data_root, f"subject {subject}", f"poses_subject{subject}_take{take}.pt")
                
                if not os.path.exists(pose_path):
                    print(f"Warning: Corresponding pose file not found for {combined_path}. Skipping.")
                    continue

                # Load the small pose file to get the length
                pose_data = torch.load(pose_path, weights_only=True)
                # The original files store samples in a list, so we use len()
                num_samples_in_file = len(pose_data)

                # IMPORTANT: We store the path to the LARGE combined file for use in __getitem__
                self.file_map.append({'path': combined_path, 'length': num_samples_in_file})
                total_samples += num_samples_in_file
                self.cumulative_samples.append(total_samples)
                del pose_data

            self._len = total_samples
            
            # Save the newly created metadata to the cache file for next time
            metadata = {'file_map': self.file_map, 'cumulative_samples': self.cumulative_samples}
            with open(metadata_cache_path, 'w') as f:
                json.dump(metadata, f)
            print(f"Metadata cache saved to: {metadata_cache_path}")

        print(f"Dataset initialized. Total samples: {self._len}")
        
        # An LRU cache for the most recently used file data
        # self.data_cache = OrderedDict()
        # self.cache_size = cache_size
        
        # Debug counters
        self.files_loaded = 0
        self.current_file_path = None
        self.current_file_data = None

    def _get_memory_usage(self):
        """Get current memory usage in GB"""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 ** 3)  # Convert to GB

    def __len__(self):
        return self._len
    
    def __getitem__(self, idx):
        if idx < 0 or idx >= self._len:
            raise IndexError("Index out of range")

        # Find which file contains the sample using the pre-computed map
        file_idx = torch.searchsorted(torch.tensor(self.cumulative_samples, dtype=torch.long), idx, right=True).item() - 1
        
        file_info = self.file_map[file_idx]
        file_path = file_info['path']
        
        # Calculate the local index within that file
        local_idx = idx - self.cumulative_samples[file_idx]

        # Simple optimization: if we're accessing the same file as last time, reuse it
        if file_path == self.current_file_path and self.current_file_data is not None:
            data = self.current_file_data
        else:
            # Clear previous file data explicitly
            if self.current_file_data is not None:
                del self.current_file_data
                self.current_file_data = None
                gc.collect()  # Force garbage collection
            
            # Memory usage before loading
            mem_before = self._get_memory_usage()
            
            # If cache is full, remove the least recently used item BEFORE loading new data
            # if len(self.data_cache) >= self.cache_size:
            #     removed_path, removed_data = self.data_cache.popitem(last=False)
            #     del removed_data  # Explicitly delete
            #     gc.collect()  # Force garbage collection
            #     print(f"Evicted {os.path.basename(removed_path)} from cache")
            
            # Load new data
            data = torch.load(file_path, weights_only=True)
            self.current_file_path = file_path
            self.current_file_data = data
            self.files_loaded += 1
            
            # Memory usage after loading
            mem_after = self._get_memory_usage()
            
            print(f"Loaded {os.path.basename(file_path)} | "
                  f"Memory: {mem_before:.1f}GB -> {mem_after:.1f}GB | "
                  f"Total files loaded: {self.files_loaded}")
            
        return {
            'image': data['image'][local_idx],
            'pose': data['pose'][local_idx],
            'pressure_map': data['pressure'][local_idx]
        }

# Test and usage example
def main():
    """Demonstrates using the MultiFile dataset with splitting."""
    # This is the directory with the large, combined triplet files
    data_dir = "/scratch/avs7793/work_done/poseembroider/new_model/src/data/combined/"
    # This is the root directory that contains the 'subject X' folders
    source_root = "/scratch/avs7793/work_done/poseembroider/new_model/src/data/"
    
    print("--- Initializing Dataset ---")
    full_dataset = MultiFilePressurePoseDataset(
        data_directory=data_dir, 
        source_data_root=source_root
    )
    print("\n--- Dataset Ready ---")

    # --- Splitting Dataset ---
    train_size = int(0.7 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    
    generator = torch.Generator().manual_seed(781)
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size], generator=generator)

    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    # --- Creating DataLoader ---
    print("\n--- Creating DataLoader (Memory-Safe Mode) ---")
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0, pin_memory=False)
    
    # Test getting a sample
    print("Fetching first batch...")
    first_train_batch = next(iter(train_loader))
    print("\n--- Verifying Batch ---")
    print(f"First train batch image shape: {first_train_batch['image'].shape}")
    print("Successfully created and tested the optimized data pipeline.")

if __name__ == "__main__":
    main()