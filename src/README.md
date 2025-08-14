data 
the data is in the format of image path, pose and pressure. Now to recreate the rsults the image paths in the data must be modified so that they can function with the pose. The image paths are stores in the format "/scratch/avs7793/work_done/poseembroider/new_model/src/data/processed/images/subject_{subject}/take_{take}/{index}.png"

now if one were to recreate all of this they should change the image paths accordingly, as the pressure and pose are all synced up with the index in the image path. 