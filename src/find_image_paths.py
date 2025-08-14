from new_model.src.data_loader.dataset import ImagePosePressureDataset
import torch
import torch.utils.data as data
import new_model.src.config as config
import os
import json
from torch import nn



def find_image_paths(dataset, combination):
	base_path = os.path.join('/scratch/avs7793/work_done/poseembroider/visualization_results', combination)
	print(f"Processing combination: {combination} from {base_path}")

	folders = [
        "best_rank_1",
        "good_rank_2_5", 
        "moderate_rank_6_10",
        "poor_rank_gt_10",
        "random",
    ]

	

	for folder in folders:
		print(f"\nProcessing folder: {folder}")

		image_paths = {}

		folder_path = os.path.join(base_path, folder)
		json_file = os.path.join(folder_path, 'summary.json')

		if not os.path.exists(json_file):
			print(f"JSON file not found: {json_file}")
			continue

		with open(json_file, 'r') as f:
			summary = json.load(f)
			
		query_idx = dataset.indices[summary['query_idx']]
		image_paths["query"] = [summary['query_idx'], dataset.dataset.get_image_path(query_idx), query_idx]

		for i in summary['top_10_indices']:
			original_idx = dataset.indices[i]
			image_path = dataset.dataset.get_image_path(original_idx)
			image_paths[i] = [image_path, original_idx]
		
		output_json_path = os.path.join(folder_path, 'image_paths.json')
		with open(output_json_path, 'w') as f:
			json.dump(image_paths, f, indent=4)
		
		print(f"Saved image paths to: {output_json_path}")
		
	return image_paths


def find_foot_pressure_mse(dataset, combination, loss_fn):
	base_path = os.path.join('/scratch/avs7793/work_done/poseembroider/visualization_results', combination)
	print(f"Processing combination: {combination} from {base_path}")

	folders = [
        "best_rank_1",
        "good_rank_2_5", 
        "moderate_rank_6_10",
        "poor_rank_gt_10",
        "random",
    ]

	

	for folder in folders:
		print(f"\nProcessing folder: {folder}")

		pressure_mse = {}

		folder_path = os.path.join(base_path, folder)
		json_file = os.path.join(folder_path, 'summary.json')

		if not os.path.exists(json_file):
			print(f"JSON file not found: {json_file}")
			continue

		with open(json_file, 'r') as f:
			summary = json.load(f)
			
		local_query_idx = summary['query_idx']
		global_query_idx = dataset.indices[local_query_idx]
		query_pressure = dataset[local_query_idx]['pressure_map']
		pressure_mse["query"] = [local_query_idx, global_query_idx]

		for local_i in summary['top_10_indices']:
			mse = loss_fn( dataset[local_i]['pressure_map'], query_pressure)
			global_i = dataset.indices[local_i]
			pressure_mse[local_i] = [mse.item(), global_i]
		
		output_json_path = os.path.join(folder_path, 'pressure_mse.json')
		with open(output_json_path, 'w') as f:
			json.dump(pressure_mse, f, indent=4)
		
		print(f"Saved image paths to: {output_json_path}")
		
	return pressure_mse




if __name__ == "__main__":
	full_dataset = ImagePosePressureDataset(metadata_file=config.DATA_FILE)
	test_size = int(0.2 * len(full_dataset))
	train_size = len(full_dataset) - test_size
	generator = torch.Generator().manual_seed(config.SEED)
	_, test_dataset = data.random_split(full_dataset, [train_size, test_size], generator=generator)
	combinations = [
		"image_to_pose",
		"image_to_pressure",
		"pose_to_image",
		"pose_to_pressure",
		"pressure_to_image",
		"pressure_to_pose",
		"image+pressure_to_pose",
		"pose+pressure_to_image",
		"image+pose_to_pressure",
	]
	loss_fn = nn.MSELoss(reduction='mean')
	for combination in combinations:
		# image_paths = find_image_paths(test_dataset, combination)
		pressure_mse = find_foot_pressure_mse(test_dataset, combination, loss_fn)

	# print(image_paths)