"""
image_proj_model.py

This module defines the ImageProjModel class, which is responsible for
projecting image embeddings into a different dimensional space. The model 
leverages a linear transformation followed by a layer normalization to 
reshape and normalize the input image embeddings for further processing in 
cross-attention mechanisms or other downstream tasks.

Classes:
    ImageProjModel

Dependencies:
    torch
    diffusers.ModelMixin

"""

import torch
from diffusers import ModelMixin


class EnvProjModel(ModelMixin):
    """
    ImageProjModel is a class that projects image embeddings into a different
    dimensional space. It inherits from ModelMixin, providing additional functionalities
    specific to image projection.

    Attributes:
        cross_attention_dim (int): The dimension of the cross attention.
        clip_embeddings_dim (int): The dimension of the CLIP embeddings.
        clip_extra_context_tokens (int): The number of extra context tokens in CLIP.

    Methods:
        forward(image_embeds): Forward pass of the ImageProjModel, which takes in image
        embeddings and returns the projected tokens.

    """

    def __init__(
        self,
        input_dim=1024,
        latent_dim=768,
    ):
        super().__init__()

        self.proj = torch.nn.Linear(
            input_dim, latent_dim
        )
        self.norm = torch.nn.LayerNorm(latent_dim)

    def forward(self, image_embeds):
        """
        Forward pass of the ImageProjModel, which takes in image embeddings and returns the
        projected tokens after reshaping and normalization.

        Args:
            image_embeds (torch.Tensor): The input image embeddings, with shape
            batch_size x num_image_tokens x clip_embeddings_dim.

        Returns:
            clip_extra_context_tokens (torch.Tensor): The projected tokens after reshaping
            and normalization, with shape batch_size x (clip_extra_context_tokens *
            cross_attention_dim).

        """
        embeds = image_embeds
        clip_extra_context_tokens = self.proj(embeds)
        clip_extra_context_tokens = self.norm(clip_extra_context_tokens)
        return clip_extra_context_tokens
