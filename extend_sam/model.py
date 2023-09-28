import torch
import torch.nn as nn
from torch.nn import functional as F

from .segment_anything import sam_model_registry


class PneuSam(nn.Module):
    def __init__(
        self, ckpt_path=None, model_type='vit_b',
    ):
        super().__init__()
        sam_model = sam_model_registry[model_type](
            checkpoint=ckpt_path)
        self.image_encoder = sam_model.image_encoder
        self.mask_decoder = sam_model.mask_decoder
        # for param in self.image_encoder.parameters():
        #     param.requires_grad = False
        self.prompt_encoder = sam_model.prompt_encoder
        # freeze prompt encoder
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False

    def forward(self, image, box):  # (B, 256, 64, 64)
        # do not compute gradients for prompt encoder
        with torch.no_grad():
            image_embedding = self.image_encoder(image)
            box_torch = torch.as_tensor(
                box, dtype=torch.float32, device=image.device)
            if len(box_torch.shape) == 2:
                box_torch = box_torch[:, None, :]  # (B, 1, 4)
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=None,
                boxes=box_torch,
                masks=None,
            )

        low_res_masks, _ = self.mask_decoder(
            image_embeddings=image_embedding,  # (B, 256, 64, 64)
            image_pe=self.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
            multimask_output=False,
        )
        ori_res_masks = F.interpolate(
            low_res_masks,
            size=(image.shape[2], image.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        return ori_res_masks
