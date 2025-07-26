import torch
import os
import glob
from tqdm import tqdm
import logging

# --- Configuration ---
# Root directory for all processed data from preprocess.py
PROCESSED_DATA_ROOT = "/scratch/avs7793/work_done/poseembroider/new_model/src/data/processed"
# Root directory for the original pressure data
PRESSURE_DATA_ROOT = "/scratch/avs7793/work_done/poseembroider/new_model/src/data/pressure"
# Output file for the final metadata
METADATA_OUTPUT_FILE = os.path.join(PROCESSED_DATA_ROOT)

# Define which subjects and takes to process.
# IMPORTANT: THIS SHOULD MATCH THE MAPPING IN `preprocess.py`
SUBJECT_TAKE_MAPPING = {
    # "1": range(1, 10),  # Process Subject 1, takes 1 through 9
    # "2": range(1, 7),   # Process Subject 5, takes 1 through 8 take 3 has less poses
    # "3": range(1, 10),
    # "4": range(1, 12),
    # "5": range(1, 9),
    # "6": range(1, 11),
    # "7": range(1, 10),
    # "8": range(1, 11),
    # "9": range(1, 14),
    "10": range(1, 11),

    # 2,7 3,10  4,12  5,9  6,11   7,10  8,11  9,14  10, 11
    # Add other subjects and their take ranges here
}

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_triplets():
    """
    Scans processed data directories to create a metadata file containing
    (image_path, pose_tensor, pressure_tensor) triplets for the entire dataset.
    """
    all_samples = []
    logging.info("Starting triplet creation process...")

    # Create a list of all takes to process for a better progress bar
    tasks = []
    for subject_id, take_range in SUBJECT_TAKE_MAPPING.items():
        for take_id in take_range:
            tasks.append((subject_id, take_id))

    for subject_id, take_id in tqdm(tasks, desc="Processing all takes"):
        # --- Define paths for the current take ---
        image_dir = os.path.join(PROCESSED_DATA_ROOT, "images", f"subject_{subject_id}", f"take_{take_id}")
        pose_path = os.path.join(PROCESSED_DATA_ROOT, "poses", f"poses_subject{subject_id}_take{take_id}.pt")
        
        # The pressure path uses the original, unprocessed data structure format
        pressure_path = os.path.join(PRESSURE_DATA_ROOT, f"pressure_subject{subject_id}_take{take_id}.pt")

        # --- Validate that all data sources exist ---
        if not os.path.isdir(image_dir) or not os.path.exists(pose_path) or not os.path.exists(pressure_path):
            logging.warning(f"Data missing for Subject {subject_id}, Take {take_id}. Skipping.")
            continue

        # --- Load data ---
        image_paths = sorted(glob.glob(os.path.join(image_dir, "*.png")))
        poses = torch.load(pose_path)
        pressure_maps = torch.load(pressure_path)

        # --- Synchronize by finding the minimum length ---
        min_len = min(len(image_paths), len(poses), len(pressure_maps))
        if not (len(image_paths) == len(poses) == len(pressure_maps)):
            logging.warning(
                f"Mismatched lengths for S{subject_id} T{take_id}: "
                f"Imgs({len(image_paths)}), Poses({len(poses)}), Pressure({len(pressure_maps)}). "
                f"Using min length: {min_len}"
            )

        # --- Create and append samples ---
        for i in range(min_len):
            sample = {
                'image_path': image_paths[i],
                'pose': poses[i],
                'pressure': pressure_maps[i]
            }
            all_samples.append(sample)

    if not all_samples:
        logging.error("No samples were created. Please check paths and data in configuration.")
        return

    # --- Save the final metadata list ---
    logging.info(f"Saving {len(all_samples)} synchronized samples to {METADATA_OUTPUT_FILE}")
    os.makedirs(os.path.dirname(METADATA_OUTPUT_FILE), exist_ok=True)
    torch.save(all_samples, os.path.join(METADATA_OUTPUT_FILE, f"triplets_metadata_subject{subject_id}.pt"))
    logging.info("--- Metadata creation complete. ---")

if __name__ == "__main__":
    create_triplets()