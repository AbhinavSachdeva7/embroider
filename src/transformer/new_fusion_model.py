import os
import torch
import torch.nn as nn
from functools import partial

# We'll need the loss function from the original repository for the final step
from new_model.src.loss_functions import symBBC, BBC


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


class L2Norm(nn.Module):
	def forward(self, x):
		return x / x.norm(dim=-1, keepdim=True)


class AddModule(nn.Module):

	def __init__(self, axis=0):
		super(AddModule, self).__init__()
		self.axis = axis

	def forward(self, x):
		return x.sum(self.axis)


class ConCatModule(nn.Module):

	def __init__(self):
		super(ConCatModule, self).__init__()

	def forward(self, x):
		x = torch.cat(x, dim=1)
		return x


class FusionTransformer(nn.Module):
	def __init__(self, latent_dim: int = 512, n_layers=4, n_heads=4, ff_dim=1024, l2normalize=False, no_projection_heads=False):
		super().__init__()

		# --- Part 1: Core Transformer Attributes ---
		self.latent_dim = latent_dim
		self.l2normalize = l2normalize

		# 1. Learnable tokens (I've renamed 'press_token_emb' to 'pressure_token_emb' for clarity)
		self.global_token = nn.Parameter(torch.zeros(1, 1, latent_dim))
		self.image_token_emb = nn.Parameter(torch.zeros(1, 1, latent_dim))
		self.pose_token_emb = nn.Parameter(torch.zeros(1, 1, latent_dim))
		self.pressure_token_emb = nn.Parameter(torch.zeros(1, 1, latent_dim))

		# 2. Transformer encoder
		encoder_layer = nn.TransformerEncoderLayer(
			d_model=latent_dim,
			nhead=n_heads,
			dim_feedforward=ff_dim,
			activation="gelu",
			batch_first=True,
		)
		self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
		self.norm = nn.LayerNorm(latent_dim)
		
		# --- Part 2: Output Projection Heads & Loss (NEW) ---
		
		# 3. Output projection heads to predict modality embeddings from the fused token
		if no_projection_heads:
			self.modality_projection_image = nn.Sequential(L2Norm()) if self.l2normalize else nn.Sequential()
			self.modality_projection_pose = nn.Sequential(L2Norm()) if self.l2normalize else nn.Sequential()
			self.modality_projection_pressure = nn.Sequential(L2Norm()) if self.l2normalize else nn.Sequential()
		else:
			self.modality_projection_image = MiniMLP(self.latent_dim, output_dim=self.latent_dim, normalize=self.l2normalize)
			self.modality_projection_pose = MiniMLP(self.latent_dim, output_dim=self.latent_dim, normalize=self.l2normalize)
			self.modality_projection_pressure = MiniMLP(self.latent_dim, output_dim=self.latent_dim, normalize=self.l2normalize)

		# 4. Temperature for scaling the contrastive loss
		self.temperature = torch.nn.Parameter( torch.FloatTensor((10,)) )

		# 5. Initialize weights
		self.init_weights()


	def init_weights(self):
		"""Initializes the learnable global token."""
		nn.init.normal_(self.global_token, std=1e-6)


	def forward_transformer_core(self, v=None, p=None, s=None):
		"""
		This is your original forward method. It takes projected modality embeddings
		and returns the single fused inter-modality token 'z'.
		v, p, s are assumed to be of shape (B, 1, latent_dim).
		"""
		# Determine batch size from the first available input
		if v is not None: B = v.size(0)
		elif p is not None: B = p.size(0)
		elif s is not None: B = s.size(0)
		else: raise ValueError("At least one modality must be provided.")

		# Inject modality-specific encodings
		inputs = [self.global_token.expand(B, -1, -1)]
		if v is not None:
			inputs.append(v + self.image_token_emb)
		if p is not None:
			inputs.append(p + self.pose_token_emb)
		if s is not None:
			inputs.append(s + self.pressure_token_emb)

		x = torch.cat(inputs, dim=1)
		x = self.transformer(x)
		x = self.norm(x)
		z = x[:, 0]
		return z


	def project_for_contrastive_losses(self, x, x_ref_dict, loss_type, prefix=""):
		"""
		NEW: Projects the fused token 'x' back into each modality's space and
		computes the contrastive loss against the original reference embeddings.
		"""
		l_dict = {}
		for k_ref in x_ref_dict:
			if x_ref_dict[k_ref] is not None:
				# k_ref is 'image_emb', k_proj is 'image'
				k_proj = k_ref.replace('_emb', '')
				
				# Get the appropriate projection head (e.g., self.modality_projection_image)
				projection_head = getattr(self, f'modality_projection_{k_proj}')
				x_proj = projection_head(x)

				# Compute scores and loss
				scores = x_proj.mm(x_ref_dict[k_ref].t())
				l_dict[prefix+k_proj] = eval(loss_type)(scores * self.temperature)
		return l_dict


	def forward(self, v=None, p=None, s=None,
					single_partials=False, dual_partials=True, triplet_partial=True,
					loss_type="symBBC",
					**kwargs):
		"""
		NEW: Main forward pass that orchestrates the full training logic,
		calculating losses for various partial inputs. This is what your
		training loop will call.
		"""
		
		# 1. Store original embeddings as reference for the loss function.
		# The loss is computed on (B, D) tensors, so we squeeze the token dimension.
		x_ref_dict = {
			'image_emb': v.squeeze(1) if v is not None else None,
			'pose_emb': p.squeeze(1) if p is not None else None,
			'pressure_emb': s.squeeze(1) if s is not None else None,
		}

		# 2. Create all partial input combinations and get their fused tokens.
		x_dict = {}
		if single_partials:
			if v is not None: x_dict['only_image_input'] = self.forward_transformer_core(v=v)
			if p is not None: x_dict['only_pose_input'] = self.forward_transformer_core(p=p)
			if s is not None: x_dict['only_pressure_input'] = self.forward_transformer_core(s=s)
		if dual_partials:
			if v is not None and p is not None: x_dict['missing_pressure_input'] = self.forward_transformer_core(v=v, p=p)
			if v is not None and s is not None: x_dict['missing_pose_input'] = self.forward_transformer_core(v=v, s=s)
			if p is not None and s is not None: x_dict['missing_image_input'] = self.forward_transformer_core(p=p, s=s)
		if triplet_partial:
			if v is not None and p is not None and s is not None: x_dict['full_input'] = self.forward_transformer_core(v=v, p=p, s=s)

		# 3. For each fused token, project it back and compute contrastive losses.
		loss_dict = {}
		for k in x_dict:  # k is e.g. 'only_image_input'
			l = self.project_for_contrastive_losses(x_dict[k], x_ref_dict, loss_type, prefix=f'{k}_clossw_')
			loss_dict.update(l)


		return loss_dict
	
	def get_retrieval_features(self, v=None, p=None, s=None):
		"""
		Runs the fusion transformer for inference to get projected features.
		
		This method is used for retrieval. It takes modality embeddings,
		fuses them, and then projects the fused embedding into each
		modality's space using the output projection heads.

		Returns:
			dict: A dictionary where keys are like 'full_input' or 'only_image_input'
				  and values are dictionaries of projected features,
				  e.g., {'image': proj_feat, 'pose': proj_feat, 'pressure': proj_feat}.
		"""
		# 1. Create all partial input combinations and get their fused tokens.
		# This reuses the exact same logic as your `forward` method.
		x_dict = {}
		# Single-modality fusions
		if v is not None: x_dict['only_image_input'] = self.forward_transformer_core(v=v)
		if p is not None: x_dict['only_pose_input'] = self.forward_transformer_core(p=p)
		if s is not None: x_dict['only_pressure_input'] = self.forward_transformer_core(s=s)
		# Dual-modality fusions
		if v is not None and p is not None: x_dict['image_pose_input'] = self.forward_transformer_core(v=v, p=p)
		if v is not None and s is not None: x_dict['image_pressure_input'] = self.forward_transformer_core(v=v, s=s)
		if p is not None and s is not None: x_dict['pose_pressure_input'] = self.forward_transformer_core(p=p, s=s)
		# Triplet-modality fusion
		if v is not None and p is not None and s is not None: x_dict['full_input'] = self.forward_transformer_core(v=v, p=p, s=s)

		# 2. For each fused token, project it into all three target modality spaces.
		retrieval_features = {}
		for fusion_key, fused_token in x_dict.items():
			projected_features = {}
			# Project to image space
			projected_features['image'] = self.modality_projection_image(fused_token)
			# Project to pose space
			projected_features['pose'] = self.modality_projection_pose(fused_token)
			# Project to pressure space
			projected_features['pressure'] = self.modality_projection_pressure(fused_token)
			
			retrieval_features[fusion_key] = projected_features
		
		return retrieval_features
	
