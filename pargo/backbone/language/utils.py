# Copyright (c) 2024 Bytedance Ltd.
# SPDX-License-Identifier: BSD-3-Clause
import torch
import torch.nn as nn
from pargo.utils import is_url, download_cached_file
from collections import OrderedDict
from copy import deepcopy
import os


__all__ = ['LayoutEmbeddings']


class LayoutEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, encoder_width):
        super(LayoutEmbeddings, self).__init__()
        self.x_position_embeddings = nn.Embedding(1024, encoder_width)
        self.y_position_embeddings = nn.Embedding(1024, encoder_width)
        self.h_position_embeddings = nn.Embedding(1024, encoder_width)
        self.w_position_embeddings = nn.Embedding(1024, encoder_width)
        self.LayerNorm = nn.LayerNorm(encoder_width)
        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(2 * encoder_width, 2)

    def forward(
        self,
        bbox=None,
        inputs_embeds=None,
    ):
        if bbox is None:
            return inputs_embeds
        words_embeddings = inputs_embeds
        try:
            left_position_embeddings = self.x_position_embeddings(bbox[:, :, 0])
            upper_position_embeddings = self.y_position_embeddings(bbox[:, :, 1])
            right_position_embeddings = self.x_position_embeddings(bbox[:, :, 2])
            lower_position_embeddings = self.y_position_embeddings(bbox[:, :, 3])
        except IndexError as e:
            raise IndexError("The :obj:`bbox`coordinate values should be within 0-1000 range.") from e

        h_position_embeddings = self.h_position_embeddings(bbox[:, :, 3] - bbox[:, :, 1])
        w_position_embeddings = self.w_position_embeddings(bbox[:, :, 2] - bbox[:, :, 0])
        
        # embeddings = (
        #     words_embeddings
        #     + left_position_embeddings
        #     + upper_position_embeddings
        #     + right_position_embeddings
        #     + lower_position_embeddings
        #     + h_position_embeddings
        #     + w_position_embeddings
        # )

        layout_embedding = (left_position_embeddings
            + upper_position_embeddings
            + right_position_embeddings
            + lower_position_embeddings
            + h_position_embeddings
            + w_position_embeddings)
        importance = self.linear(torch.cat([words_embeddings, layout_embedding], dim=-1)).sigmoid()
        text_importance = torch.unsqueeze(importance[..., 0], dim=2)
        layout_importance = torch.unsqueeze(importance[..., 1], dim=2)
        
        embeddings = words_embeddings * text_importance + layout_embedding * layout_importance

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings




def load_checkpoint(model, url_or_filename:str, no_missing=False, remove_prefix=''):
    
    if not url_or_filename: return
    
    if is_url(url_or_filename):
        cached_file = download_cached_file(url_or_filename, progress=True)
        checkpoint = torch.load(cached_file, map_location='cpu') 
    elif url_or_filename.startswith('hdfs:'):
        checkpoint = torch.load(os.path.basename(url_or_filename), map_location='cpu')
    elif os.path.isfile(url_or_filename):
        checkpoint = torch.load(url_or_filename, map_location='cpu')
    else:
        raise RuntimeError('checkpoint url or path is invalid')
        
    if 'model' in checkpoint.keys():    # *.pth file
        state_dict = checkpoint['model']
    elif 'state_dict' in checkpoint.keys(): # *.ckpt file
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    if remove_prefix != '':
        remove_prefix = remove_prefix + '.'
        new_sd = OrderedDict()
        for k,v in state_dict.items():
            if k.startswith(remove_prefix):
                removed_name = k[len(remove_prefix):]
                new_sd[removed_name] = v
            else:
                new_sd[k] = v
        state_dict = deepcopy(new_sd)
        del new_sd
    
    for key in model.state_dict().keys():
        if key in state_dict.keys():
            if state_dict[key].shape!=model.state_dict()[key].shape:
                print(f"Del: {key} Shape: model={model.state_dict()[key].shape} ckpt={state_dict[key].shape}")
                del state_dict[key]
    
    msg = model.load_state_dict(state_dict,strict=False)
    print(f'load checkpoint from {url_or_filename}')
    # print(msg)
    if no_missing:
        assert(len(msg.missing_keys)==0)
    # return model