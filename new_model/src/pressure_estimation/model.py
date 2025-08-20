import torch
import torch.nn as nn
from new_model.src.encoder.base_encoders.base_pressure_encoder import LargeVariationalAutoEncoder
from new_model.src.full_model import PoseImagePressureEmbroider
import new_model.src.config as config

# this dies not work only here as proof of concept
class PressureEstimator(nn.Module):
    def __init__(self, latentD=512):
        super(PressureEstimator, self).__init__()
        self.latentD = latentD

        self.pressure_encoder = LargeVariationalAutoEncoder()
        for p in self.pressure_encoder.parameters():
            p.requires_grad = False
        self.pressure_encoder.load_state_dict(torch.load(config.PRESSURE_ENCODER_PATH, map_location=torch.device('cpu')))
        

        self.fusion_transformer = PoseImagePressureEmbroider(latentD=512)
        for p in self.fusion_transformer.parameters():
            p.requires_grad = False
        checkpoint = torch.load(config.FUSION_TRANSFORMER_PATH, map_location=torch.device('cpu'))
        self.fusion_transformer.load_state_dict(checkpoint['model_state_dict'])
        
        self.first_layer = nn.Linear(latentD, 256)
        self.second_layer = nn.Linear(256, 64)
        self.ReLU = nn.ReLU()

    def forward(self, pose):
        fused_embedding = self.fusion_transformer.get_fused_embedding(poses=pose)
        z = fused_embedding['only_pose_input']
        z = self.ReLU(self.first_layer(z))
        z = self.second_layer(z) # 64
        z_decoded = self.pressure_encoder.decode(z)
        return z_decoded
    

class PressureEstimatorNew(nn.Module):
    def __init__(self, latentD=512):
        super(PressureEstimatorNew, self).__init__()
        self.latentD = latentD

        self.pressure_encoder = LargeVariationalAutoEncoder()
        for p in self.pressure_encoder.parameters():
            p.requires_grad = False
        self.pressure_encoder.load_state_dict(torch.load(config.PRESSURE_ENCODER_PATH, map_location=torch.device('cpu')))
        

        self.fusion_transformer = PoseImagePressureEmbroider(latentD=512)
        for p in self.fusion_transformer.parameters():
            p.requires_grad = False
        checkpoint = torch.load(config.FUSION_TRANSFORMER_PATH, map_location=torch.device('cpu'))
        self.fusion_transformer.load_state_dict(checkpoint['model_state_dict'])
        
        self.first_layer = nn.Linear(latentD, 2*latentD)
        self.second_layer = nn.Linear(2*latentD, 256)
        self.third_layer = nn.Linear(256, 64)
        self.ReLU = nn.ReLU()

    def forward(self, pose=None, image=None):
        fused_embedding = self.fusion_transformer.get_fused_embedding(poses=pose, images=image)
        
        if image is not None and pose is not None:
            z = fused_embedding['missing_pressure_input']
        elif image is not None and pose is None:
            z = fused_embedding['only_image_input']
        elif pose is not None and image is None:
            z = fused_embedding['only_pose_input']
        
        z = self.ReLU(self.first_layer(z)) #  512 -> 1024
        z = self.ReLU(self.second_layer(z)) # 1024 -> 256
        z = self.third_layer(z) # 256 -> 64
        z_decoded = self.pressure_encoder.decode(z)
        return z_decoded