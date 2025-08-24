# PoseImagePressureEmbroider: A Multi-Modal Retrieval Model

This project implements a deep learning model for cross-modal retrieval between human images, SMPL poses, and tactile pressure maps. The core of the model is a fusion transformer that learns a shared embedding space for these three modalities. This allows for querying the dataset with one or more modalities to retrieve corresponding data from another.

Additionally, this repository includes a downstream application for estimating pressure maps directly from pose and/or image data.

## Key Features

- **Multi-Modal Embedding**: Learns a shared latent space for images, poses, and pressure data.
- **Cross-Modal Retrieval**:
  - **Single-Modality Query**: Retrieve corresponding data using a single input (e.g., Image → Pose, Pose → Pressure).
  - **Dual-Modality Query**: Combine two modalities to retrieve the third (e.g., Image + Pose → Pressure).
- **Downstream Task**: A dedicated model for pressure map estimation trained on the learned embeddings.

## Setup

### Prerequisites

- **Environment**: This code was developed and tested in a `conda` environment.
- **Python**: `3.9`
- **PyTorch**: `2.7`

Other required packages are mentioned in requirements.txt. These should be installed to be compatible with the Python and PyTorch versions above. For more information the conda environment package list is also included. 

### Required Data & Pre-trained Models

1.  **Dataset**: This project uses TMM-100 containing synchronized videos, motion capture data, and pressure maps. The motion-capture data is converted to smpl poses, and video is converted to images by saving each frame, these images are saved in RGB and resized to (256,192), the code in `dataset.py` expects the images to be resized for conversion to tensors. We have a data file [https://pennstateoffice365.sharepoint.com/:u:/r/sites/LPAC-LaboratoryforPerceptionActionandCognition/Shared%20Documents/Retrieval_Model_Data/ALL_DATA.pt?csf=1&web=1&e=598kFl], where the data is stored as [image_path, pose, pressure_map] the pose and pressure*map are saved as tensors, so they can be used unchanged. The image_path is saved as `/scratch/avs7793/work_done/poseembroider/new_model/src/data/processed/images/subject*{subject (1- 10)}/take\_{take number}/{index( in 5 decimal places like 00123 )}`, this needs to be changed depending on where you save the images.

2.  **SMPL Models**: they are required only if you want to visualize them, or find MPJPE.

3.  **Pre-trained Encoder Weights**: The individual encoders for image, pose, and pressure are initialized with pre-trained weights. Download these and ensure the paths in `src/config.py` point to them correctly. They can be downloaded from here. [https://pennstateoffice365.sharepoint.com/:f:/r/sites/LPAC-LaboratoryforPerceptionActionandCognition/Shared%20Documents/Retrieval_Model_Data/pretrained_weights?csf=1&web=1&e=rGJM2r]

## Run all the code from the parent directory of new_model

## Data Preparation Pipeline

The raw dataset must be processed into a format suitable for efficient training. Follow these steps sequentially:

1.  **Step 1: Process Raw Data**

    - Configure the source paths (`SOURCE_VIDEO_ROOT`, `SOURCE_POSE_ROOT`) and output directory (`OUTPUT_ROOT`) in `src/data_loader/preprocess.py`.
    - Run the script to extract video frames and process pose files:
      ```bash
      python -m new_model.src.data_loader.preprocess
      ```

2.  **Step 2: Create Data Triplets**

    - This script synchronizes the processed frames, poses, and raw pressure data.
    - Configure the `PROCESSED_DATA_ROOT` and `PRESSURE_DATA_ROOT` in `src/data_loader/make_triplets.py`.
    - Update the `SUBJECT_TAKE_MAPPING` to specify which subjects and takes to include.
    - Run the script:
      ```bash
      python -m src.data_loader.make_triplets
      ```

3.  **Step 3: Consolidate Metadata**
    - To enable fast data loading, this script combines all individual metadata files into a single large file.
    - Verify the paths in `src/data_loader/make_single_file.py`.
    - Run the script:
      ```bash
      python -m new_model.src.data_loader.make_single_file
      ```
    - This will generate the final `ALL_DATA.pt` file, which is used by the main dataset loader. This file can also be downloaded and the image_paths changed for use.

## Usage

### Configuration

Before running any script, review `src/config.py`. This file contains critical paths for datasets, pre-trained weights, and checkpoint directories. Ensure all paths are correct for your system.

### Training the Retrieval Model

To train the main PoseImagePressureEmbroider model, run the training loop:

```bash
python -m new_model.src.train_loop
```

Checkpoints will be saved periodically to the directory specified in `config.py`.

Another thing to remmeber is that the file `train_loop.py` contains the resume checkpoint flag (as a constant) and the path to the checkpoint to resume training from, if you want to train the model from the beginning disable one or both of them to ensure the model starts training from the start. 

### Benchmarking

To evaluate a trained model and generate performance metrics, run the benchmark script:

```bash
python -m new_model.src.benchmark
```

**Important**: You must edit `src/benchmark.py` and set the `CHECKPOINT_PATH` variable to point to the model checkpoint you wish to evaluate.

### Training the Pressure Estimator

To train the downstream pressure estimation model:

```bash
python -m new_model.src.pressure_estimation.train
```

### Inference and Visualization

- **Qualitative Retrieval Results**: To visualize how well the model retrieves samples across modalities, use `inference2.py`.
  ```bash
  python -m new_model.src.inference2
  ```
- **Pressure Estimation Demo**: To visualize the output of the pressure estimator for a given sample, use `visualize.py`.
  ```bash
  python -m new_model.src.pressure_estimation.visualize
  ```

## Project Structure
```markdown
src/
├── benchmark/ # Benchmark logs and results
├── checkpoints/ # Saved model checkpoints
├── config.py # Central configuration for paths and parameters
├── data_loader/ # Scripts for data preprocessing and loading
│ ├── dataset.py # Main PyTorch Dataset class
│ ├── preprocess.py # Step 1: Processes raw video and poses
│ ├── make_triplets.py # Step 2: Creates (image, pose, pressure) triplets
│ └── make_single_file.py # Step 3: Consolidates data for fast loading
├── encoder/ # Modality-specific encoders
├── full_model.py # Main PoseImagePressureEmbroider model definition
├── inference2.py # Script for qualitative retrieval visualization
├── loss_functions.py # Loss functions for training
├── pressure_estimation/ # Code for the downstream pressure estimation task
│ ├── model.py
│ ├── train.py
│ ├── test.py
│ └── visualize.py
├── train_loop.py # Main training script for the retrieval model
└── transformer/ # Fusion transformer implementation
```
