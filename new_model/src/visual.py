import matplotlib.pyplot as plt
from PIL import Image
import os
import json

# --- 1. SETUP: Define your image paths ---

# -- Replace these with the actual paths to your image files --


def visualize_results(combination):
    base_path = f'/scratch/avs7793/work_done/poseembroider/visualization_results/{combination}'
    folders = [
        "best_rank_1",
        "good_rank_2_5",
        "moderate_rank_6_10",
        "poor_rank_gt_10",
        "random",
    ]

    for folder_name in folders:
        print(f"Processing folder: {folder_name}")

        current_folder_path = os.path.join(base_path, folder_name)

        # The single query image
        if combination == "image+pose_to_pressure":
            query_image = [os.path.join(current_folder_path, 'query_image.png'), os.path.join(current_folder_path, 'smplx_pose_solid_render_3d_view_1.png')]
        elif combination == "image+pressure_to_pose":
            query_image = [os.path.join(current_folder_path, 'query_image.png'), os.path.join(current_folder_path, 'query_pressure.png')]
        elif combination == "pose+pressure_to_image":
            query_image = [os.path.join(current_folder_path, 'smplx_pose_solid_render_3d_view_1.png'), os.path.join(current_folder_path, 'query_pressure.png')]
        elif combination == "pressure_to_image" or combination == "pressure_to_pose":
            query_image = [os.path.join(current_folder_path, 'query_pressure.png')]
        elif combination == "image_to_pose" or combination == "image_to_pressure":
            query_image = [os.path.join(current_folder_path, 'query_image.png')]
        elif combination == "pose_to_image" or combination == "pose_to_pressure":
            query_image = [os.path.join(current_folder_path, 'smplx_pose_solid_render_3d_view_1.png')]
        else:
            raise ValueError(f"Invalid combination: {combination}")

        # The three ground truth images
        ground_truth_images = {
            "Image": os.path.join(current_folder_path, 'query_image.png'),
            "Pressure": os.path.join(current_folder_path, 'query_pressure.png'),
            "Pose": os.path.join(current_folder_path, 'smplx_pose_solid_render_3d_view_1.png')
        }

        # A list of file paths for the 10 ranked results
        
        target_modality = combination.split('_to_')[-1]
        img_paths_filename = 'image_paths.json'
        pressure_mse_filename = 'pressure_mse.json'
        mpjpe_filename = 'mpjpe_summary.json'
        
        img_path_file = os.path.join(current_folder_path, img_paths_filename)

        if target_modality == "pressure":
            result_image_filename = 'pressure.png'
            pressure_mse_file = os.path.join(current_folder_path, pressure_mse_filename)
        elif target_modality == "pose":
            result_image_filename = 'smplx_pose_solid_render_3d_view_1.png'
            mpjpe_file = os.path.join(base_path, mpjpe_filename)
        elif target_modality == "image":
            result_image_filename = 'image.png'
        else:
            raise ValueError(f"Could not determine target modality from combination: {combination}")

        ranked_result_images = [os.path.join(current_folder_path, f'rank_{i:02d}', result_image_filename) for i in range(1, 11)]


        with open(img_path_file, 'r') as f:
            img_path_summary = json.load(f)

        keys = list(img_path_summary.keys())
        gt_img_detail = ' '.join(img_path_summary[keys[0]][1].split('/')[-3:]).replace('_', '-').replace('.png', '')
        # --- 2. PLOTTING: Create the 4x12 grid visualization ---

        # Create a figure with a 4-row, 12-column grid.
        # The figsize is set to be wide to accommodate all columns.
        fig, axes = plt.subplots(nrows=4, ncols=12, figsize=(24, 8))

        # --- Row 1: Add Text Headings ---
        # First, turn off the axis frames for the entire first row
        for i in range(12):
            axes[0, i].axis('off')

        # Define heading properties
        heading_font = {'fontsize': 14, 'fontweight': 'bold', 'ha': 'center', 'va': 'center'}

        # Add text to the cells in the first row
        axes[0, 0].text(0.5, 0.5, "Ground Truth", **heading_font)
        axes[0, 1].text(0.5, 0.5, "Query", **heading_font)
        for i in range(10):
            axes[0, i + 2].text(0.5, 0.5, f"Rank {i+1}", **heading_font)


        # --- Column 1 (Rows 2-4): Plot Ground Truth Images ---
        gt_image_ax = axes[1, 0]
        gt_pressure_ax = axes[2, 0]
        gt_pose_ax = axes[3, 0]

        gt_image_ax.imshow(Image.open(ground_truth_images["Image"]))
        gt_image_ax.set_title("Image", fontsize=12)
        gt_image_ax.set_xlabel(f"{gt_img_detail}", fontsize=8)

        gt_pressure_ax.imshow(Image.open(ground_truth_images["Pressure"]))
        gt_pressure_ax.set_title("Pressure", fontsize=12)

        gt_pose_ax.imshow(Image.open(ground_truth_images["Pose"]))
        gt_pose_ax.set_title("Pose", fontsize=12)


        # --- Column 2 (Row 2): Plot Query Image ---
        query_ax = axes[1, 1]
        query_ax.imshow(Image.open(query_image[0]))
        if len(query_image) > 1:
            query_ax1 = axes[2,1]
            query_ax1.imshow(Image.open(query_image[1]))


        # --- Columns 3-12: Plot Ranked Results (Retrieved and Ground Truth) ---
        for i, retrieved_path in enumerate(ranked_result_images):
            # Plot retrieved result in row 2
            retrieved_ax = axes[1, i + 2]
            if os.path.exists(retrieved_path):
                retrieved_ax.imshow(Image.open(retrieved_path))
                retrieved_ax.set_title("Retrieved", fontsize=10)
                
                
                img_detail = ' '.join(img_path_summary[keys[i+1]][0].split('/')[-3:]).replace('_', '-').replace('.png', '')
                # retrieved_ax.set_xlabel(f"{img_detail}", fontsize=6)
                label_text = img_detail
                label_fontsize = 8
                if target_modality == "pose":
                    with open(mpjpe_file, 'r') as f:
                        mpjpe_summary = json.load(f)
                    mpjpe = mpjpe_summary[folder_name][f'rank_{i+1:02d}']
                    metric_text = f"MPJPE: {mpjpe:.2f} mm"
                    label_text = f"{metric_text}\n{img_detail}"
                    label_fontsize = 8
                    
                elif target_modality == "pressure":
                    with open(pressure_mse_file, 'r') as f:
                        pressure_mse_summary = json.load(f)
                    pressure_keys = list(pressure_mse_summary.keys())
                    pressure_mse = pressure_mse_summary[pressure_keys[i+1]][0]
                    metric_text = f"MSE: {pressure_mse:.2f}"
                    label_text = f"{metric_text}\n{img_detail}"
                    label_fontsize = 8

                retrieved_ax.set_xlabel(label_text, fontsize=label_fontsize)
                # elif target_modality == "image":
                #     with open(img_path_file, 'r') as f:
                #         img_path_summary = json.load(f)
            else:
                print(f"Image not found, skipping: {retrieved_path}")
                retrieved_ax.text(0.5, 0.5, "N/A", ha='center', va='center')

            # Plot ground truth for the result in row 4
            gt_ax = axes[3, i + 2]
            if os.path.exists(retrieved_path):
                if target_modality == "pressure":
                    gt_ax.imshow(Image.open(ground_truth_images["Pressure"]))
                elif target_modality == "pose":
                    gt_ax.imshow(Image.open(ground_truth_images["Pose"]))
                elif target_modality == "image":
                    gt_ax.imshow(Image.open(ground_truth_images["Image"]))
                else:
                    raise ValueError(f"Could not determine target modality from combination: {combination}")
                gt_ax.set_title("Ground Truth", fontsize=10)
            else:
                # No need to print again, just mark as N/A
                gt_ax.text(0.5, 0.5, "N/A", ha='center', va='center')


        # --- 3. Final Touches & Cleanup ---

        # Turn off all axis ticks and frames for a cleaner look
        for ax in axes.flat:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)

        # Specifically hide the unused axes to make the layout clean
        # All of row 1 is used for headings (axes are already off)
        # Hide unused cells in rows 3 and 4
        axes[2, 1].axis('off') # Hide cell below/between query images
        for col in range(2, 12):
             axes[2, col].axis('off') # Hide the entire 3rd row for ranked results columns

        # Adjust layout to prevent titles from overlapping and give some space
        head = combination.replace('_', ' ')
        plt.tight_layout(pad=2.0)
        fig.suptitle(f"{head} Retrieval Results - {folder_name}", fontsize=22, weight='bold')
        plt.subplots_adjust(top=0.90) # Adjust top to make space for the suptitle


        # --- 4. Save the Final Image ---
        output_filename = f"retrieval_results_grid_{combination}_{folder_name}.png"
        output_path = os.path.join('/scratch/avs7793/work_done/poseembroider/new_grids', output_filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')

        print(f"Result image '{output_filename}' has been saved!")
        plt.close(fig)


if __name__ == "__main__":
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
    for combination in combinations:
        visualize_results(combination)