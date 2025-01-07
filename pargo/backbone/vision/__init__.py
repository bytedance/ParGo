# Copyright (c) 2024 Bytedance Ltd.
# SPDX-License-Identifier: BSD-3-Clause
import os
import torch

from pargo.backbone.vision.eva_vit import EVA_ViT
from pargo.backbone.vision.vit import interpolate_pos_embed
from pargo.utils import download_cached_file, XHUB


__all__ = ['init_vision_model']



def init_vision_model(name:str, image_size:int, pretrained:bool=False, visiom_model="", **kwargs):
    """
    Create vision model, either from existing or custom configs.
    """
    # using preloaded weights
    if pretrained:
        root = kwargs.get('root', './')
        # cached_file = os.path.join(root, os.path.basename(XHUB['vision'][name]))
        cached_file = visiom_model
        print(f'[loading] using pretrained {cached_file} for {name}')
        # cached_file = download_cached_file(SUPPORTED_VISION_MODELS[name], progress=False)
    
    if name == 'eva_vit_g':
        """Checkpoint loaded from https://github.com/salesforce/LAVIS """
        vision_model = EVA_ViT(
            img_size=image_size, patch_size=14, embed_dim=1408, 
            depth=39, num_heads=16, mlp_ratio=4.3637,
            use_mean_pooling=False, qkv_bias=True,
            **kwargs
        )
        if pretrained:
            state_dict = torch.load(cached_file, map_location='cpu')
            if 'pos_embed' in state_dict:
                state_dict['pos_embed'] = interpolate_pos_embed(state_dict['pos_embed'], vision_model)
            msg = vision_model.load_state_dict(state_dict, strict=False)
            print(msg)
    
    else:
        raise NotImplementedError
    
    return vision_model
