import torch
import torch.nn as nn

# --- Re-used from original project ---
# from poseembroider.model_modules import ImageEncoder, PoseEncoder, get_projection
from new_model.src.encoder.encoders import PressureEncoder, ImageEncoder, PoseEncoder

# --- Your new Fusion Transformer ---
from new_model.src.transformer.new_fusion_model import FusionTransformer


class PoseImagePressureEmbroider(nn.Module):
    """
    The main container model. It holds the encoders and the fusion transformer.
    This class is responsible for the full data flow, from raw data to loss.
    """
    def __init__(self,
                 latentD=512,
                 # You can expose other params from FusionTransformer here if needed
                 **kwargs):
        super().__init__()
        
        # 1. Initialize Encoders (reusing from the original project)
        self.image_encoder = ImageEncoder(latentD=latentD)
        self.pose_encoder = PoseEncoder(latentD=latentD)
        self.pressure_encoder = PressureEncoder(latentD=latentD)

        # 2. Freeze the pretrained parts of the encoders
        self.image_encoder.set_pretrained_weights_grad(False)
        self.pose_encoder.set_pretrained_weights_grad(False)
        self.pressure_encoder.set_pretrained_weights_grad(False)

        # 3. Initialize Fusion Transformer
        self.fusion_transformer = FusionTransformer(latent_dim=latentD, **kwargs)

    def forward(self, images=None, poses=None, pressure_maps=None, **kwargs):
        
        # --- Step 1: Get projected embeddings from encoders ---
        # Note: The projection is handled *inside* each encoder module.
        image_emb = self.image_encoder(images) if images is not None else None
        pose_emb = self.pose_encoder(poses) if poses is not None else None
        pressure_emb = self.pressure_encoder(pressure_maps) if pressure_maps is not None else None
        
        # --- Step 2: Pass embeddings to the fusion transformer to get losses ---
        loss_dict = self.fusion_transformer(v=image_emb, p=pose_emb, s=pressure_emb, **kwargs)

        return loss_dict 
    
    def get_retrieval_features(self, images=None, poses=None, pressure_maps=None):
        """
        Gets retrieval features for given inputs. For inference.
        """
        image_emb, pose_emb, pressure_emb = None, None, None
        
        # Get embeddings from the respective encoders if data is provided
        with torch.no_grad(): # No need to track gradients for inference
            if images is not None:
                image_emb = self.image_encoder(images)
            if poses is not None:
                pose_emb = self.pose_encoder(poses)
            if pressure_maps is not None:
                pressure_emb = self.pressure_encoder(pressure_maps)
        
        # Pass embeddings to the fusion transformer's retrieval method
        retrieval_features = self.fusion_transformer.get_retrieval_features(v=image_emb, p=pose_emb, s=pressure_emb)
        return retrieval_features
    
