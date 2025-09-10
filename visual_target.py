import matplotlib.pyplot as plt
from PIL import Image
import os
import json

# --- Configuration ---
NUMBER_OF_RETRIEVED_RESULTS_DISPLAYED = 3

def visualize_target_results(sample_index, combination, input_base_directory='/scratch/avs7793/work_done/poseembroider/new_model/target_results', output_directory='/scratch/avs7793/work_done/poseembroider/new_model/inference_results/target_grids'):
    """
    Visualize retrieval results for target_results structure.
    
    Args:
        sample_index: The sample index (e.g., 140, 5159, etc.)
        combination: The combination string (e.g., "image_to_pose")
        input_base_directory: Base directory containing target_results
        output_directory: Directory to save visualization grids
    """
    sample_folder = f'sample_{sample_index:05d}'
    base_path = os.path.join(input_base_directory, sample_folder)
    combo_path = os.path.join(base_path, combination)
    
    print(f"Processing sample: {sample_folder}, combination: {combination}")
    
    # Check if combination directory exists
    if not os.path.exists(combo_path):
        print(f"Combination directory not found: {combo_path}")
        return
    
    # Define ground truth image paths
    ground_truth_images = {
        "Image": os.path.join(base_path, 'query_image.png'),
        "Pressure": os.path.join(base_path, 'query_pressure.png'),
        "Pose": os.path.join(base_path, '3d_view_1.png')  # From SMPLX visualization
    }
    
    # Determine query modality and target modality
    if combination.startswith("image_to_"):
        query_modality = "Image"
        target_modality = combination.split('_to_')[-1]
    elif combination.startswith("pose_to_"):
        query_modality = "Pose"
        target_modality = combination.split('_to_')[-1]
    elif combination.startswith("pressure_to_"):
        query_modality = "Pressure"
        target_modality = combination.split('_to_')[-1]
    elif combination.startswith("image+pose_to_"):
        query_modality = "Image+Pose"
        target_modality = "pressure"
    elif combination.startswith("image+pressure_to_"):
        query_modality = "Image+Pressure"
        target_modality = "pose"
    elif combination.startswith("pose+pressure_to_"):
        query_modality = "Pose+Pressure"
        target_modality = "image"
    else:
        raise ValueError(f"Invalid combination: {combination}")
    
    target_modality = target_modality.capitalize()
    
    # Get file paths and load metadata
    summary_file = os.path.join(combo_path, 'summary.json')
    img_path_file = os.path.join(combo_path, 'img_path.json')
    
    # Check required files exist
    if not os.path.exists(summary_file):
        print(f"Summary file not found: {summary_file}")
        return
    
    with open(summary_file, 'r') as f:
        summary = json.load(f)
    
    # Try to load image paths if available
    img_paths = {}
    if os.path.exists(img_path_file):
        with open(img_path_file, 'r') as f:
            img_paths = json.load(f)
    
    # Load metrics if available
    pressure_rmse_file = os.path.join(combo_path, 'pressure_rmse.json')
    mpjpe_file = os.path.join(combo_path, 'mpjpe_summary.json')
    
    pressure_metrics = {}
    mpjpe_metrics = {}
    
    if target_modality.lower() == "pressure" and os.path.exists(pressure_rmse_file):
        with open(pressure_rmse_file, 'r') as f:
            pressure_metrics = json.load(f)
    
    if target_modality.lower() == "pose" and os.path.exists(mpjpe_file):
        with open(mpjpe_file, 'r') as f:
            mpjpe_metrics = json.load(f)
    
    # Get retrieved result paths
    ranked_result_images = []
    ranked_result_pressure = []
    ranked_result_pose = []
    
    for i in range(1, NUMBER_OF_RETRIEVED_RESULTS_DISPLAYED + 1):
        rank_dir = os.path.join(combo_path, f'rank_{i:02d}')
        ranked_result_images.append(os.path.join(rank_dir, 'image.png'))
        ranked_result_pressure.append(os.path.join(rank_dir, 'pressure.png'))
        ranked_result_pose.append(os.path.join(rank_dir, '3d_view_1.png'))  # From SMPLX visualization
    
    # All indices are frames for subject 7, take 3
    subject_info = "subject 7"
    take_info = "take 3"
    frame_info = f"frame {sample_index:05d}"
    
    # Get pose name from indices dictionary  
    indices = {140:   "Preparation",
               5159:"Step Back, Reeling Forearms (repulse Monkey) 2nd Right",
               5463:"Step Back, Reeling Forearms (repulse Monkey) 2nd Left",
               5601:"Grasp Sparrow's Tail on Left, Hold the Ball",
               5800:"Grasp Sparrow's Tail on Left, Ward off (peng)",
               6305:"Grasp Sparrow's Tail on Left, Squeeze",
               6667:"Grasp Sparrow's Tail on Left, Push",
               726:"Opening",
               7187:"Grasp Sparrow's Tail on Right, Ward off (peng)",
               7390:"Grasp Sparrow's Tail on Right, Stroke",
               9866:"Cloud Hands, 3rd",
               10103:"Single Whip",
               10104:"Single Whip",
               1167:"Part the Wild Horse's Mane, 1st Left",
               10473:"Stroke Horse's Back",
               11147:"Right Heel Kick",
               11501:"Double Bees Bussing at Ears (strike opponent's ears with two fists)",
               2445 : "White Crane Spreads Its Wings",
               }
    
    pose_name = indices.get(sample_index, "Unknown Pose")
    
    # --- Create the visualization ---
    # 4x4 grid: Headers + 3 modalities (Image, Pressure, Pose)
    fig, axes = plt.subplots(nrows=4, ncols=1+NUMBER_OF_RETRIEVED_RESULTS_DISPLAYED, 
                            figsize=(16, 12))
    
    # --- Row 0: Add Text Headings ---
    for i in range(1+NUMBER_OF_RETRIEVED_RESULTS_DISPLAYED):
        axes[0, i].axis('off')
    
    heading_font = {'fontsize': 14, 'fontweight': 'bold', 'ha': 'center', 'va': 'center'}
    
    # Ground Truth header
    axes[0, 0].text(0.5, 0.5, f"Ground Truth\n{subject_info}, {take_info}\n{frame_info}", **heading_font)
    
    # Retrieved results headers with their subject/take/frame info
    for i in range(NUMBER_OF_RETRIEVED_RESULTS_DISPLAYED):
        # Get subject/take/frame info for this retrieved result
        retrieved_subject = "subject 7"  # Default
        retrieved_take = "take 3"        # Default  
        retrieved_frame = f"frame {sample_index:05d}"  # Default
        
        if img_paths and 'top_3_full_dataset_indices' in summary and i < len(summary['top_3_full_dataset_indices']):
            idx_key = str(summary['top_3_full_dataset_indices'][i])
            if idx_key in img_paths:
                img_path = img_paths[idx_key]
                path_parts = img_path.split('/')
                if len(path_parts) >= 3:
                    subject_part = path_parts[-3].replace('subject_', 'subject ')
                    take_part = path_parts[-2].replace('take_', 'take ')
                    frame_part = path_parts[-1].replace('.png', '').replace('frame_', 'frame ')
                    retrieved_subject = subject_part
                    retrieved_take = take_part
                    retrieved_frame = frame_part
        
        header_text = f"Rank {i+1}\nRetrieved\n{retrieved_subject}, {retrieved_take}\n{retrieved_frame}"
        axes[0, i + 1].text(0.5, 0.5, header_text, fontsize=12, fontweight='bold', ha='center', va='center')
    
    # --- Determine layout order: Query/Target modality on top, others below ---
    
    # For ground truth: query modality on top
    if query_modality == "Image":
        gt_order = ["Image", "Pressure", "Pose"]
        query_row = 1  # Image row
    elif query_modality == "Pose":
        gt_order = ["Pose", "Image", "Pressure"] 
        query_row = 1  # Pose row
    elif query_modality == "Pressure":
        gt_order = ["Pressure", "Image", "Pose"]
        query_row = 1  # Pressure row
    else:  # Multi-modal queries - put primary modality first
        if "Image+Pose" in query_modality:
            gt_order = ["Image", "Pose", "Pressure"]
            query_row = 1
        elif "Image+Pressure" in query_modality:
            gt_order = ["Image", "Pressure", "Pose"]
            query_row = 1
        else:
            gt_order = ["Pressure", "Pose", "Image"] 
            query_row = 1
    
    # For retrieved results: target modality on top
    if target_modality.lower() == "image":
        retrieved_order = ["Image", "Pressure", "Pose"]
        target_row = 1
    elif target_modality.lower() == "pose":
        retrieved_order = ["Pose", "Image", "Pressure"]
        target_row = 1
    elif target_modality.lower() == "pressure":
        retrieved_order = ["Pressure", "Image", "Pose"]
        target_row = 1
    
    # Map modalities to their paths
    gt_modality_paths = {
        "Image": ground_truth_images["Image"],
        "Pressure": ground_truth_images["Pressure"],
        "Pose": ground_truth_images["Pose"]
    }
    
    retrieved_modality_paths = {
        "Image": ranked_result_images,
        "Pressure": ranked_result_pressure,
        "Pose": ranked_result_pose
    }
    
    # --- Plot all 3 modalities ---
    for row_idx in range(3):  # 3 modalities
        actual_row = row_idx + 1  # +1 because row 0 is headers
        
        # Get modalities for this row
        gt_modality = gt_order[row_idx]
        retrieved_modality = retrieved_order[row_idx]
        
        # Ground truth
        gt_ax = axes[actual_row, 0]
        gt_path = gt_modality_paths[gt_modality]
        
        if os.path.exists(gt_path):
            gt_ax.imshow(Image.open(gt_path))
            # Only mark query modality, remove other titles
            if actual_row == query_row:
                gt_ax.set_title(f"{gt_modality}\n(Query)", fontsize=14, fontweight='bold', color='red')
        else:
            gt_ax.text(0.5, 0.5, f"{gt_modality}\nN/A", ha='center', va='center')
        
        # Retrieved results for this modality
        retrieved_paths = retrieved_modality_paths[retrieved_modality]
        
        for i, retrieved_path in enumerate(retrieved_paths):
            retrieved_ax = axes[actual_row, i + 1]
            
            if os.path.exists(retrieved_path):
                retrieved_ax.imshow(Image.open(retrieved_path))
                
                # Only mark target modality, remove other titles
                if actual_row == target_row and i == 0:
                    retrieved_ax.set_title(f"{retrieved_modality}\n(Target)", fontsize=14, fontweight='bold', color='blue')
                
                # Add metric labels for target modality only, positioned closer to image
                if actual_row == target_row:
                    label_text = ""
                    if target_modality.lower() == "pose" and mpjpe_metrics:
                        rank_key = f'rank_{i+1:02d}'
                        if rank_key in mpjpe_metrics:
                            mpjpe_val = mpjpe_metrics[rank_key]
                            label_text = f"MPJPE: {mpjpe_val:.2f} mm"
                    elif target_modality.lower() == "pressure" and pressure_metrics:
                        if 'top_3_full_dataset_indices' in summary and i < len(summary['top_3_full_dataset_indices']):
                            idx_key = str(summary['top_3_full_dataset_indices'][i])
                            if idx_key in pressure_metrics:
                                rmse_val = pressure_metrics[idx_key][0]
                                label_text = f"RMSE: {rmse_val:.2f}"
                    
                    if label_text:
                        # Position closer to image
                        retrieved_ax.text(0.5, -0.08, label_text, ha='center', va='top', 
                                        transform=retrieved_ax.transAxes, fontsize=10, fontweight='bold')
            else:
                retrieved_ax.text(0.5, 0.5, "N/A", ha='center', va='center')
    
    # --- Final Touches ---
    # Turn off all axis ticks and frames
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
    
    # Create enhanced title
    query_target_text = f"{query_modality}(Query) -> {target_modality}(Target)"
    main_title = f"{subject_info}, {take_info}, {frame_info} - {pose_name} - {query_target_text}"
    
    plt.tight_layout(pad=1.0)
    fig.suptitle(main_title, fontsize=18, weight='bold')
    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.05, right=0.95, 
                       hspace=0.25, wspace=0.1)
    
    # Save the result
    os.makedirs(output_directory, exist_ok=True)
    output_filename = f"retrieval_results_{sample_folder}_{combination}.png"
    output_path = os.path.join(output_directory, output_filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    print(f"Visualization saved: {output_filename}")
    plt.close(fig)

def visualize_all_target_results():
    """Visualize all available target results."""
    # Configuration from img_path.py
    indices = {140:   "Preparation",
               5159:"Step Back, Reeling Forearms (repulse Monkey) 2nd Right",
               5463:"Step Back, Reeling Forearms (repulse Monkey) 2nd Left",
               5601:"Grasp Sparrow's Tail on Left, Hold the Ball",
               5800:"Grasp Sparrow's Tail on Left, Ward off (peng)",
               6305:"Grasp Sparrow's Tail on Left, Squeeze",
               6667:"Grasp Sparrow's Tail on Left, Push",
               726:"Opening",
               7187:"Grasp Sparrow's Tail on Right, Ward off (peng)",
               7390:"Grasp Sparrow's Tail on Right, Stroke",
               9866:"Cloud Hands, 3rd",
               10103:"Single Whip",
               10104:"Single Whip",
               1167:"Part the Wild Horse's Mane, 1st Left",
               10473:"Stroke Horse's Back",
               11147:"Right Heel Kick",
               11501:"Double Bees Bussing at Ears (strike opponent's ears with two fists)",
               2445 : "White Crane Spreads Its Wings",
               }
    



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
    
    input_base_directory = '/scratch/avs7793/work_done/poseembroider/new_model/target_results'
    output_directory = '/scratch/avs7793/work_done/poseembroider/new_model/inference_results/target_grids'
    
    for index in indices.keys():
        for combination in combinations:
            try:
                visualize_target_results(
                    sample_index=index,
                    combination=combination,
                    input_base_directory=input_base_directory,
                    output_directory=output_directory
                )
            except Exception as e:
                print(f"Error processing sample {index}, combination {combination}: {e}")
                continue

if __name__ == "__main__":
    # You can run all or specific ones
    visualize_all_target_results()
    
    # Or run specific samples/combinations:
    # visualize_target_results(sample_index=140, combination="image_to_pose")