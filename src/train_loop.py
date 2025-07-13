import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from new_model.src.full_model import PoseImagePressureEmbroider
import new_model.src.config as config
from new_model.src.data_loader.dataset import MultiFilePressurePoseDataset

# personally do not like passing arguments through command line, so any and all parameters to be passed are in config.py
def train():
    LEARNING_RATE = 1e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    torch.manual_seed(config.SEED)
    torch.cuda.manual_seed(config.SEED)
    LOG_STEP = 10
    SAVE_EPOCH_INTERVAL = 5
    EPOCHS = 5
    BATCH_SIZE = 4
    NUM_WORKERS = 0

    # image transforms for the image encoder, temporary will move this to dataset module
    image_transform = transforms.Compose([
        transforms.Resize((256, 192)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # load the data
    train_dataset = MultiFilePressurePoseDataset(data_directory=config.TRAIN_DATA_DIR, source_data_root=config.SOURCE_DATA_ROOT)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

    model = PoseImagePressureEmbroider(latentD=config.LATENT_D).to(device)
    
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)

    print("Starting training...")
    for epoch in range(EPOCHS):
        model.train()
        total_epoch_loss = 0

        for i, batch in enumerate(train_loader):
            # Move data to the correct device
            images = batch['image'].to(device)
            poses = batch['pose'].to(device)
            pressure_maps = batch['pressure_map'].to(device)

            # Zero the gradients from the last step
            optimizer.zero_grad()
            
            # --- FORWARD PASS ---
            # Call the model to get the dictionary of all contrastive losses
            loss_dict = model(images=images, poses=poses, pressure_maps=pressure_maps, 
                              single_partials=True, dual_partials=True, triplet_partial=True)
            
            if not loss_dict:
                continue
            
            # Aggregate the losses from all partials
            loss = sum(loss_dict.values()) / len(loss_dict)

            # --- BACKWARD PASS ---
            loss.backward()
            optimizer.step()
            
            total_epoch_loss += loss.item()

            if (i + 1) % LOG_STEP == 0:
                print(f'Epoch [{epoch+1}/{EPOCHS}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

        avg_epoch_loss = total_epoch_loss / len(train_loader)
        print(f"--- End of Epoch [{epoch+1}/{EPOCHS}], Average Loss: {avg_epoch_loss:.4f} ---")       

        if (epoch + 1) % SAVE_EPOCH_INTERVAL == 0:
            torch.save(model.state_dict(), f'/scratch/avs7793/work_done/poseembroider/new_model/src/checkpoints/model_epoch_{epoch+1}.pth')



if __name__ == "__main__":
    train()



