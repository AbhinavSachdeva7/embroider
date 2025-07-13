import json

# --- File Paths ---
# TODO: You must update these paths to point to where you will store the pretrained models.



# The posevae_models.json file should look like this:
# { "posevae_model_bedlamscript": "/path/to/your/posevae_model_bedlamscript.pth" }
IMAGE_ENCODER_PATH = f"/scratch/avs7793/work_done/poseembroider/new_model/src/pretrained_weights/smpler_x_b32.pth.tar"
POSE_ENCODER_PATH = f"/scratch/avs7793/work_done/poseembroider/new_model/src/pretrained_weights/checkpoint_release.pth"
PRESSURE_ENCODER_PATH = f"/scratch/avs7793/work_done/poseembroider/new_model/src/pretrained_weights/model_epoch_300.pt"

SEED = 781

LATENT_D = 512
# --- Model Parameters ---
NB_INPUT_JOINTS = 22 # From the original PoseEmbroider config

TRAIN_DATA_DIR = "/scratch/avs7793/work_done/poseembroider/new_model/src/data/combined/"
SOURCE_DATA_ROOT = "/scratch/avs7793/work_done/poseembroider/new_model/src/data/"


# --- Utility Functions ---
def read_json(absolute_filepath):
	with open(absolute_filepath, 'r') as f:
		data = json.load(f)
	return data
