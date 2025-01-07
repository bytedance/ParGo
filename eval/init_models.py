# Copyright (2024) Bytedance Ltd.
# SPDX-License-Identifier: BSD-3-Clause
import os
from unittest import result
import numpy as np
from PIL import Image
from loguru import logger
import torch
import torch.nn as nn
from torchvision import transforms
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch.utils.data as data
from pargo.backbone.vision import init_vision_model
from pargo.backbone.vision.clip_vit import LayerNorm
from pargo.backbone.vision.eva_vit import create_eva_vit
from pargo.backbone.fusion.minigpt import disabled_train, maybe_autocast, convert_weights_to_fp16, encode_img, embed_tokens
import pandas as pd
import argparse
from pargo.backbone.fusion.blip2 import init_ParGo
def split_into_patches(tensor, patch_size=224):
    """
    Split a batch of images into patches.
    
    Args:
    - tensor: A tensor of shape (B, C, H, W) where B is the batch size,
      C is the number of channels, H is the height, and W is the width.
    - patch_size: The size of the patches (default 224).
    
    Returns:
    - A list of patch tensors, each of shape (B, C, patch_size, patch_size).
    """
    
    B, C, H, W = tensor.size()
    patches = []
    
    # Calculate the number of patches to split along height and width
    num_patches_height = H // patch_size
    num_patches_width = W // patch_size
    
    for i in range(num_patches_height):
        for j in range(num_patches_width):
            # Define the start and end indices for height and width
            start_i = i * patch_size
            end_i = start_i + patch_size
            start_j = j * patch_size
            end_j = start_j + patch_size
            
            # Extract the patch tensor using slicing
            patch = tensor[:, :, start_i:end_i, start_j:end_j]
            patches.append(patch)
    
    return patches

class MiniGPT_v2(nn.Module):
    def __init__(self,
        vision_model_name="eva_vit_g",
        image_size=224,
        vision_model="<Your Directory>/eva_vit_g.pth",
        vit_precision="fp16",
        vit_freeze=True,
        llm_model="<Your Directory>/7B-chat",
        pad_sym="$$",
        llm_freeze=True,
        lora_r=0,
        lora_alpha=16,
        lora_dropout=0.05,
        lora_target_modules=["q_proj","k_proj","v_proj"],
        fusion_method="linear",
        ckpt_path="",    # trained model
        visual_tokens_adjacent=4,
        bert_type=None,
        num_query_token=None,
        num_local_query_token=0,
        cross_attention_freq=None,
    ):
        super().__init__()
        # vision model
        if not "eva-clip" in vision_model_name:
            self.visual_encoder = init_vision_model(
                name=vision_model_name, 
                image_size=image_size, 
                pretrained=True,
                visiom_model=vision_model,
                drop_path_rate=0,
                use_checkpoint=False,
            )
        else:
            eva_clip_params = {
                "embed_dim": 768,
                "img_size": 336,
                "layers": 24,
                "width": 1024,
                "drop_path_rate": 0,
                "head_width": 64,
                "mlp_ratio": 2.6667,
                "patch_size": 14,
                "eva_model_name": "eva-clip-l-14-336",
                "xattn": True,
                "fusedLN": True,
                "rope": True,
                "pt_hw_seq_len": 16,
                "intp_freq": True,
                "naiveswiglu": True,
                "subln": True,
            }
            self.visual_encoder = create_eva_vit(**eva_clip_params)
            print("vision model:", vision_model)
        
        self.ln_vision = LayerNorm(self.visual_encoder.num_features)
        if vit_precision == "fp16":
            print("Convert vision model to fp16.")
            convert_weights_to_fp16(self.visual_encoder)
        if vit_freeze:
            print("Freeze vision model.")
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            for name, param in self.ln_vision.named_parameters():
                param.requires_grad = False
            self.ln_vision = self.ln_vision.eval()
            self.ln_vision.train = disabled_train
        
        # language model
        print("Load llm tokenizer, ", llm_model)
        self.llm_tokenizer = AutoTokenizer.from_pretrained(
            llm_model,
            use_fast=False,
            trust_remote_code=True,
            model_max_length=1024
        )
        if self.llm_tokenizer.pad_token is None:
            self.llm_tokenizer.pad_token = pad_sym
            print("llm_tokenizer.pad_token is None, to use:", self.llm_tokenizer.pad_token)
            print("llm_tokenizer.pad_token_id:", self.llm_tokenizer.pad_token_id)
        print("Load llm model, ", llm_model)
        self.llm_model = AutoModelForCausalLM.from_pretrained(
            llm_model,
            torch_dtype=torch.float16,
            trust_remote_code=True
            # load_in_8bit=True,
            # device_map={'': 0},
        )
        if llm_freeze:
            print("Freeze llm model.")
            for name, param in self.llm_model.named_parameters():
                param.requires_grad = False
        # LoRA
        if lora_r > 0:
            print("LoRA for llm model.")
            self.llm_model = prepare_model_for_kbit_training(self.llm_model)
            loraconfig = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=lora_target_modules,
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.llm_model = get_peft_model(self.llm_model, loraconfig)
            self.llm_model.print_trainable_parameters()
        
        # fusion method
        if fusion_method == "linear":
            self.llm_proj = nn.Linear(
                self.visual_encoder.num_features * visual_tokens_adjacent, self.llm_model.config.hidden_size
            )
        elif fusion_method == "mlp":
            self.llm_proj = nn.Sequential(
                nn.Linear(self.visual_encoder.num_features * visual_tokens_adjacent, self.llm_model.config.hidden_size),
                nn.GELU(),
                nn.Linear(self.llm_model.config.hidden_size, self.llm_model.config.hidden_size),
            )
        elif fusion_method == "ParGo":
            self.ParGo, self.query_tokens = init_ParGo(
                bert_type,
                num_query_token,
                num_local_query_token,
                self.visual_encoder.num_features, 
                cross_attention_freq
            )
            self.llm_proj = nn.Linear(
                self.ParGo.config.hidden_size, self.llm_model.config.hidden_size
            )
            self.ParGo.cls = None
            self.ParGo.bert.embeddings.word_embeddings = None
            for layer in self.ParGo.bert.encoder.layer:
                layer.output = None
                layer.intermediate = None
        # load params
        if ckpt_path:
            print("Load fusion model, ckpt_path:", ckpt_path)
            self.load_fusion_model(ckpt_path)
    
    def _encode_img(self,visual_encoder, ln_vision, llm_proj, image, query_tokens,window_size=4,text_embs=None):
        """use vit model get image embedding"""
        with maybe_autocast():
            image_embeds = ln_vision(visual_encoder(image, return_all_features=True)).to(image.device) # eva_vit_g is [1, 257, 1408], eva-clip-l-14-336 is [1, 577, 1024]
            image_embeds = image_embeds[:, 1:, :]
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
            query_tokens = query_tokens.expand(image_embeds.shape[0], -1, -1)
            if text_embs is not None:
                query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(image.device)
                Qformer_atts = torch.cat([query_atts, text_embs.attention_mask],dim=1).to(image.device)
                query_output = self.ParGo.bert(
                    text_embs.input_ids,
                    attention_mask=Qformer_atts,
                    query_embeds=query_tokens,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_atts,
                    use_cache=True,
                    return_dict=True,
                )
            else:
                query_output = self.ParGo.bert(
                    query_embeds=query_tokens,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_atts,
                    use_cache=True,
                    return_dict=True,
                )
            inputs_llama = self.llm_proj(query_output.last_hidden_state)
            atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(image.device)         
        return inputs_llama, atts_llama
    def new_encode_img(self,visual_encoder, ln_vision, llm_proj, image, query_tokens,window_size=4):
        """use vit model get image embedding"""
        with maybe_autocast():
            images_embeds_list=[]
            for img in split_into_patches(image, patch_size=336):
                image_embeds = ln_vision(visual_encoder(img, return_all_features=True)).to(image.device) # eva_vit_g is [1, 257, 1408], eva-clip-l-14-336 is [1, 577, 1024]
                image_embeds = image_embeds[:, 1:, :]
                images_embeds_list.append(image_embeds)
            image_embeds = torch.cat(images_embeds_list, dim=1)

            
            # image_embeds = ln_vision(visual_encoder(image))
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
            query_tokens = query_tokens.expand(image_embeds.shape[0], -1, -1)
            # print (image_embeds.shape,query_tokens.shape)
            query_output = self.ParGo.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                use_cache=True,
                return_dict=True,
            )
            
            inputs_llama = self.llm_proj(query_output.last_hidden_state)
            atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(image.device)         
        return inputs_llama, atts_llama

    def context_embeds(self, images, prompts, max_context_len, device, window_size=4,fusion_method=None):
        """prepare image_embeds and prompt embeds to inputs_embeds"""
        if fusion_method == "ParGo":
            images_embs, images_atts = self.new_encode_img(self.visual_encoder, self.ln_vision,self.ParGo, images.to(device), self.query_tokens)
        else:
            images_embs, images_atts = encode_img(self.visual_encoder, self.ln_vision, self.llm_proj, images.to(device), window_size)# for pretrain inference, bos + image_beddings
        if prompts is None or len(prompts) == 0:
            bos_token = self.llm_tokenizer(["<s>"] * images_embs.shape[0], return_tensors="pt", add_special_tokens=False).to(device)
            bos_emb = embed_tokens(self.llm_model, bos_token.input_ids)
            embs = torch.cat([bos_emb, images_embs], dim=1)
            attn_mask = torch.cat([bos_token.attention_mask, images_atts], dim=1)
            return embs, attn_mask

        # concat bos、image、prompt embedding
        batch_embs = []
        for idx, prompt in enumerate(prompts):
            # print (prompts)
            prompt_segs = prompt.split('<ImageHere>')
            seg_tokens = [
                self.llm_tokenizer(seg, return_tensors="pt", add_special_tokens=False).to(device).input_ids
                for i, seg in enumerate(prompt_segs)
            ]

            seg_embs = [embed_tokens(self.llm_model, seg_t) for seg_t in seg_tokens]
            
            bos_token = self.llm_tokenizer(["<s>"], return_tensors="pt", add_special_tokens=False).to(device)
            bos_emb = embed_tokens(self.llm_model, bos_token.input_ids)
            
            mixed_embs = [bos_emb] + [seg_embs[0]] + [images_embs[idx, :, :].view(1, images_embs.shape[1], images_embs.shape[2])] + [seg_embs[-1]]
            mixed_embs = torch.cat(mixed_embs, dim=1)
            batch_embs.append(mixed_embs)

        batch_size = len(batch_embs)
        max_len = max([emb.shape[1] for emb in batch_embs])
        emb_dim = batch_embs[0].shape[2]
        dtype = batch_embs[0].dtype

        embs = torch.zeros([batch_size, max_len, emb_dim], dtype=dtype, device=device)
        attn_mask = torch.zeros([batch_size, max_len], dtype=torch.int, device=device)
        
        # left padding
        for i, emb in enumerate(batch_embs):
            emb_len = emb.shape[1]
            embs[i, -emb_len:] = emb[0]
            attn_mask[i, -emb_len:] = 1
        
        return embs, attn_mask

    def load_fusion_model(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        if "model" in ckpt: # from open source minigpt
            state_dict = {}
            for key in ckpt["model"]:
                if "llama" in key:
                    # print(key, ckpt["model"][key].shape)
                    state_dict[key.replace("llama", "llm")] = ckpt["model"][key]
            msg = self.load_state_dict(state_dict, strict=False)
            print("load open source:", msg)
        elif "state_dict" in ckpt and "step=" in ckpt_path:
            msg = self.load_state_dict(ckpt["state_dict"], strict=False)
        elif "pytorch_model.bin" in ckpt_path: # from unit deepspeed
            state_dict = {}
            for key in ckpt:
                state_dict[key.replace("module.", "")] = ckpt[key]
            msg = self.load_state_dict(state_dict, strict=False)
            print("load unit deepspeed:", msg)
        else:
            msg = self.load_state_dict(ckpt, strict=False)
            print("load:", msg)
            