# Copyright (c) 2024 Bytedance Ltd.
# SPDX-License-Identifier: BSD-3-Clause
import torch
import torch.nn as nn
import contextlib

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

def maybe_autocast(dtype=torch.float16):
    # if on cpu, don't use autocast
    # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
    if torch.cuda.is_available():
        return torch.cuda.amp.autocast(dtype=dtype)
    else:
        return contextlib.nullcontext()

def convert_weights_to_fp16(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def convert_weights_to_fp16_(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

    model.apply(convert_weights_to_fp16_)

def encode_img(visual_encoder, ln_vision, llm_proj, image, window_size=4):
    """use vit model get image embedding"""
    device = image.device
    if len(image.shape) > 4:
        image = image.reshape(-1, *image.shape[-3:])
    with maybe_autocast():
        image_embeds = ln_vision(visual_encoder(image, return_all_features=True)).to(device) # eva_vit_g is [1, 257, 1408], eva-clip-l-14-336 is [1, 577, 1024]
        image_embeds = image_embeds[:, 1:, :]
        bs, pn, hs = image_embeds.shape
        image_embeds = image_embeds.view(bs, int(pn / window_size), int(hs * window_size))

        inputs_llama = llm_proj(image_embeds)
        atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(image.device)            
    return inputs_llama, atts_llama

def concat_emb_input_output(input_embs, input_atts, output_embs, output_atts):
    """concat image、prompt and label token embedding, includer attention_mask"""
    input_lens = []
    cat_embs = []
    cat_atts = []
    for i in range(input_embs.size(0)):
        input_len = input_atts[i].sum()
        input_lens.append(input_len)
        cat_embs.append(
            torch.cat([
                input_embs[i][:input_len],
                output_embs[i],
                input_embs[i][input_len:]
            ])
        )
        cat_atts.append(
            torch.cat([
                input_atts[i][:input_len],
                output_atts[i],
                input_atts[i][input_len:]
            ])
        )
    cat_embs = torch.stack(cat_embs)
    cat_atts = torch.stack(cat_atts)
    return cat_embs, cat_atts, input_lens

def embed_tokens(llm_model, token_ids):
    """map token_ids to token_embeddings"""
    if llm_model.base_model.config.architectures[0] == "InternLM2ForCausalLM":
        if hasattr(llm_model.base_model, "model"): # lora wrapped model
            embeds = llm_model.base_model.model.model.get_input_embeddings()(token_ids)
        else:
            embeds = llm_model.base_model.get_input_embeddings()(token_ids)
    elif hasattr(llm_model.base_model, "model"): # lora wrapped model
        embeds = llm_model.base_model.model.model.embed_tokens(token_ids)
    elif hasattr(llm_model.base_model, "wte"): # gpt2
        embeds = llm_model.base_model.wte(token_ids)
    else:
        embeds = llm_model.base_model.embed_tokens(token_ids)
    return embeds

def prompt_wrap(llm_model, llm_tokenizer, img_embeds, img_atts, prompts, lengths=None, max_context_len=3800, img_sys="<ImageHere>"):
    """
    tokenize prompt and concat image embeddings and attention_masks
    """
    if prompts is None or len(prompts) == 0:
        # prompts is not provided, just return the original image embedding
        return img_embeds, img_atts
    elif img_embeds is None:
        # prompt is provided but there is no image embedding. return the prompt embedding in right padding
        llm_tokenizer.padding_side = "right"
        prompt_tokens = llm_tokenizer(
            prompts,
            return_tensors="pt",
            padding="longest",
            add_special_tokens=False
        )
        prompt_embeds = embed_tokens(llm_model, prompt_tokens.input_ids)
        prompt_atts = prompt_tokens.attention_mask
        return prompt_embeds, prompt_atts
    else:
        # return the multi-modal embedding in right padding
        emb_lists = []
        if isinstance(prompts, str):
            prompts = [prompts] * len(img_embeds)

        for idx, (each_img_embed, each_prompt) in enumerate(zip(img_embeds, prompts)):
            pn = each_img_embed.shape[-2] # visual token num
            if lengths is not None:
                each_img_embed = each_img_embed.reshape(-1, each_img_embed.shape[-1])
                each_img_embed = each_img_embed[:lengths[idx] * pn]
            p_segs = each_prompt.split(img_sys)
            interleave_emb = []
            for idx, seg in enumerate(p_segs[:-1]):
                p_tokens = llm_tokenizer(
                    seg, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
                p_embed = embed_tokens(llm_model, p_tokens.input_ids)
                interleave_emb.append(torch.cat([p_embed, each_img_embed[None][:, idx * pn:(idx + 1) * pn]], dim=1))
            wrapped_emb = torch.cat(interleave_emb, dim=1)
            p_tokens = llm_tokenizer(
                p_segs[-1], return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
            p_embed = embed_tokens(llm_model, p_tokens.input_ids)
            wrapped_emb = torch.cat([wrapped_emb, p_embed], dim=1)
            emb_lists.append(wrapped_emb)

        emb_lens = [emb.shape[1] for emb in emb_lists]
        pad_emb = embed_tokens(llm_model, torch.tensor(llm_tokenizer.pad_token_id, device=img_embeds.device))

        max_length = max(emb_lens) if max(emb_lens) < max_context_len else max_context_len
        wrapped_embs = pad_emb.expand(len(emb_lens), max_length, -1).clone()
        wrapped_atts = torch.zeros([len(emb_lens), max_length], dtype=torch.int, device=img_embeds.device)
        
        for i, emb in enumerate(emb_lists):
            length = emb_lens[i] if emb_lens[i] < max_context_len else max_context_len
            wrapped_embs[i, :length] = emb[:, :length]
            wrapped_atts[i, :length] = 1
        return wrapped_embs, wrapped_atts