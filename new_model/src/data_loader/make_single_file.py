import torch
import os
from tqdm import tqdm
import logging
import glob

# --- Configuration ---
# The directory containing the metadata files from `make_triplets.py`
METADATA_INPUT_DIR = "/scratch/avs7793/work_done/poseembroider/new_model/src/data/processed/"
# The path for the final, consolidated data file
FINAL_OUTPUT_FILE = "/scratch/avs7793/work_done/poseembroider/new_model/src/data/processed/ALL_DATA.pt"

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def consolidate_metadata_files(input_dir: str, output_path: str):
    """
    Scans a directory for subject-specific metadata files (triplets_metadata_subject*.pt),
    combines them, and consolidates the result into a single dictionary
    with large, stacked tensors for efficiency.
    """
    search_pattern = os.path.join(input_dir, "triplets_metadata_subject*.pt")
    metadata_files = sorted(glob.glob(search_pattern))

    if not metadata_files:
        logging.error(f"No metadata files found matching pattern: {search_pattern}")
        return

    logging.info(f"Found {len(metadata_files)} metadata files to consolidate.")

    # --- Initialize master lists to hold all data from all files ---
    all_image_paths = []
    all_poses_list = []
    all_pressures_list = []

    # --- Iterate through each file and append its data ---
    for file_path in tqdm(metadata_files, desc="Consolidating metadata files"):
        samples_list = torch.load(file_path)

        if not samples_list:
            logging.warning(f"Metadata file {os.path.basename(file_path)} is empty. Skipping.")
            continue
        
        for sample in samples_list:
            all_image_paths.append(sample['image_path'])
            all_poses_list.append(sample['pose'])
            all_pressures_list.append(sample['pressure'])
    
    if not all_image_paths:
        logging.error("No samples found across all metadata files. Nothing to save.")
        return

    # --- Stack tensors for memory efficiency ---
    logging.info("Stacking all collected pose and pressure tensors...")
    final_poses = torch.stack(all_poses_list)
    final_pressures = torch.stack(all_pressures_list)
    
    # Create the final consolidated dictionary
    final_data = {
        'image_paths': all_image_paths,
        'poses': final_poses,
        'pressures': final_pressures
    }

    # --- Save the single, large data file ---
    logging.info(f"Saving final consolidated data to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(final_data, output_path)
    
    logging.info("\n--- Successfully combined all data into a single file. ---")
    logging.info(f"Final number of image paths: {len(all_image_paths)}")
    logging.info(f"Final pose tensor shape: {final_poses.shape}")
    logging.info(f"Final pressure tensor shape: {final_pressures.shape}")


if __name__ == '__main__':
    consolidate_metadata_files(input_dir=METADATA_INPUT_DIR, output_path=FINAL_OUTPUT_FILE)