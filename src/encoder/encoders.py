import torch
from torch import nn
from collections import OrderedDict
import os

import new_model.src.config as config

from new_model.src.encoder.base_encoders.base_pressure_encoder import LargeVariationalAutoEncoder
from new_model.src.encoder.base_encoders.base_pose_encoder import PoseEncoder as PoseEncoder_posevae
from new_model.src.encoder.base_encoders.base_image_encoder import ViT


class MiniMLP(nn.Module):

	def __init__(self, input_dim, hidden_dim=None, output_dim=None, normalize=False, three_layers=False):
		super(MiniMLP, self).__init__()
		
		if hidden_dim == output_dim == None:
			hidden_dim = output_dim = input_dim
		elif hidden_dim == None:
			hidden_dim = input_dim

		layers = [
				nn.Linear(input_dim, hidden_dim),
				nn.ReLU(),
		]
		if three_layers:
			layers += [
				nn.Linear(hidden_dim, hidden_dim),
				nn.ReLU(),
				nn.Dropout(0.1),
			]
		layers += [ nn.Linear(hidden_dim, output_dim) ]

		self.layers = nn.Sequential(*layers)
		self.normalize = normalize

		self.init_weights()


	def init_weights(self):
		for layer in self.layers:
			if isinstance(layer, nn.Linear):
				nn.init.trunc_normal_(layer.weight, std=0.02)
				if layer.bias is not None:
					nn.init.zeros_(layer.bias)


	def forward(self, x):
		x = self.layers(x)
		if self.normalize:
			x = nn.functional.normalize(x, dim=-1)
		return x

def average_pooling(token_embeddings, attention_mask=None):
	# take attention mask into account for correct mean pooling of all token embeddings
	batch_size, nbtokens, embed_dim = token_embeddings.shape
	if attention_mask is None: attention_mask = torch.ones(batch_size, nbtokens, device=token_embeddings.device).long()
	input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
	x = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
	return x.view(batch_size, 1, embed_dim)

def get_projection(projection_type, pretrained_output_dim, latentD):
	if projection_type=="layerplus":
		return nn.Sequential(
			nn.Linear(pretrained_output_dim, latentD),
			nn.ReLU(),
			nn.Dropout(0.1),
		)
	elif projection_type=="minimlp":
		return MiniMLP(input_dim=pretrained_output_dim, output_dim=latentD)
	else:
		raise NotImplementedError


class ImageEncoder(nn.Module):
	

	def __init__(self, latentD=512, projection_type="layerplus"):
		super(ImageEncoder, self).__init__()
		self.latentD = latentD
		
		# 1. Initialize the base ViT model
		backbone_config = dict(img_size=(256, 192), patch_size=16, embed_dim=768, depth=12, num_heads=12, ratio=1, use_checkpoint=False, mlp_ratio=4, qkv_bias=True, drop_path_rate=0.3)
		self.base_encoder = ViT(**backbone_config)
		
		# 2. Load its pretrained weights
		pretrained_path = config.IMAGE_ENCODER_PATH
		ckpt = torch.load(pretrained_path, map_location=torch.device('cpu'))
		new_state_dict = OrderedDict()
		for k, v in ckpt['network'].items():
			if ('backbone.' in k) or ('encoder.' in k):
				k = k.replace('backbone.', '').replace('encoder.', '')
				if k.startswith('module.'): k = k[len("module."):]
				new_state_dict[k] = v
		self.base_encoder.load_state_dict(new_state_dict, strict=True)
		print(f"Initialized ImageEncoder with: {pretrained_path}")
		
		# 3. Define the trainable projection head
		self.projection = get_projection(projection_type, self.base_encoder.embed_dim, self.latentD)

	def set_pretrained_weights_grad(self, value: bool):
		for param in self.base_encoder.parameters():
			param.requires_grad = value

	def forward(self, images, avgpooling_of_tokens=True):
		# Get raw features from base encoder
		raw_ft, _ = self.base_encoder.forward_features(x=images)
		raw_ft = raw_ft.flatten(2,3).permute(0, 2, 1)
		# Pass through our trainable projection
		projected_ft = self.projection(raw_ft)
		# Pool if necessary
		if avgpooling_of_tokens:
			projected_ft = average_pooling(projected_ft)
		return projected_ft

class PoseEncoder(nn.Module):

	def __init__(self, latentD=512, projection_type='layerplus'):
		super(PoseEncoder, self).__init__()
		self.latentD = latentD
		
		# 1. Load checkpoint info
		pretrained_path = config.POSE_ENCODER_PATH
		ckpt = torch.load(pretrained_path, 'cpu')
		
		# 2. Initialize the base PoseVAE model
		self.base_encoder = PoseEncoder_posevae(latentD=ckpt['args'].latentD, num_body_joints=config.NB_INPUT_JOINTS, role="no_output_layer")
		
		# 3. Load its pretrained weights
		self.base_encoder.load_state_dict({k[len('pose_encoder.'):]: v for k,v in ckpt['model'].items() if k.startswith('pose_encoder.') and 'encoder.8' not in k})
		print(f"Initialized PoseEncoder with: {pretrained_path}")
		
		# 4. Define the trainable projection head
		pretrained_output_dim = self.base_encoder.encoder[7].weight.shape[1]
		self.projection = get_projection(projection_type, pretrained_output_dim, self.latentD)

	def set_pretrained_weights_grad(self, value: bool):
		for param in self.base_encoder.parameters():
			param.requires_grad = value

	def forward(self, poses):
		raw_ft = self.base_encoder(poses)
		projected_ft = self.projection(raw_ft)
		return projected_ft.view(len(poses), 1, -1)

class PressureEncoder(nn.Module):
	"""
	A self-contained wrapper for your pretrained Foot Pressure VAE. It handles:
	1. Extracting the `mu` vector from the VAE.
	2. Projecting the 64-D `mu` vector to `latentD` with a TRAINABLE head.
	"""		
	def __init__(self, latentD=512, projection_type="layerplus"):
		super().__init__()
		
		self.pretrained_pressure_encoder = LargeVariationalAutoEncoder()
		pretrained_path = config.PRESSURE_ENCODER_PATH
		self.pretrained_pressure_encoder.load_state_dict(torch.load(pretrained_path, map_location=torch.device('cpu')))
		
		self.pressure_projection = get_projection(projection_type, 64, latentD)

	def set_pretrained_weights_grad(self, value: bool = False):
		"""Freezes or unfreezes the underlying VAE, leaves projection head trainable."""
		for p in self.pretrained_pressure_encoder.parameters():
			p.requires_grad = value	

	def forward(self, foot_pressure_map_flattened):
		"""Forward pass: VAE's encode -> trainable projection."""
	    # The VAE expects a flattened input.
	    # flat_map = foot_pressure_map.view(foot_pressure_map.size(0), -1)
	    # We only need the encoder part, so we call .encode()
		mu, _ = self.pretrained_pressure_encoder.encode(foot_pressure_map_flattened)
		projected_mu = self.pressure_projection(mu)
	    # Return in the standard (Batch, Tokens, Dim) format
		return projected_mu.view(projected_mu.size(0), 1, -1)