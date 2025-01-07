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
from pargo.data.datasets.mmebench_datasets import MMEBenchDataset
import pandas as pd
import argparse
from init_models import MiniGPT_v2
from init_models import split_into_patches
def main(args):
    model = MiniGPT_v2(
        vision_model_name=args.vision_model_name,
        vision_model=args.vision_model,
        llm_model=args.llm_model,
        image_size=args.image_size,
        pad_sym="$$",
        ckpt_path=args.ckpt,
        lora_target_modules=args.lora_target_modules,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        fusion_method=args.fusion_method,
        visual_tokens_adjacent=args.visual_tokens_adjacent,
        bert_type=args.bert_type,
        num_query_token=args.num_query_token,
        num_local_query_token = args.num_local_query_token,
        cross_attention_freq=args.cross_attention_freq,
    )
    if torch.cuda.is_available():
        device="cuda"
        model = model.cuda()
    else:
        device = "cpu"

    model = model.eval()

    # concat bos、image、prompt embedding and attention mask
    result=[]
    test_input='{}/Data_json/'.format(args.input)
    output = args.output
    if not os.path.exists(output):
        os.makedirs(output)
    root = args.input
    prompt_temp = args.prompt_template
    for filename in os.listdir(test_input):
        if os.path.exists(os.path.join(output, filename)):
            continue
        logger.info("{}, {}".format(test_input,filename))
        with open(os.path.join(output, filename), 'w') as fout:
            datax=MMEBenchDataset(data_file=os.path.join(test_input, filename), root=root, image_size=args.image_size, prompt_template=prompt_temp)
            dataloader = data.DataLoader(datax, batch_size=args.batch_size, shuffle=False)
            for i, batch in enumerate(dataloader):
                inputs_embeds, attention_mask = inputs_embeds, attention_mask = model.context_embeds(batch['img'], batch['prompts'], 3800, device, args.visual_tokens_adjacent,fusion_method=args.fusion_method)
                outputs = model.llm_model.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    max_new_tokens=100,
                    num_beams=4,
                    length_penalty=1,
                    temperature=0.9,
                    do_sample=False,
                    min_length=1,
                    top_p=0.4,
                    repetition_penalty=1,
                )
                outputs = model.llm_tokenizer.batch_decode(outputs, add_special_tokens=False)
                for idx,output_text in enumerate(outputs):
                    # if need post process, todo add
                    # print (output_text)
                    # print (filename, i*args.batch_size+idx, output_text.replace('\n',' '), batch['answer'][idx], batch['prompts'][idx].replace('\n',' '), batch['img_path'][idx])
                    output_text = output_text.split('</s>')[0]
                    output_text = output_text.replace("<s>", "")
                    output_text = output_text.split(r'[/INST]')[-1].strip() # [INST] <Img><ImageHere></Img> [vqa] {} [/INST]
                    output_text = output_text.split(r'###')[0].strip()      # <Img><ImageHere></Img> ###Human :{} ###Assistant :
                    # print ('*'*10)
                    print (batch['img_path'][idx], batch['prompts'][idx].replace('\n',' '), batch['answer'][idx], output_text.replace('\n',' ').replace('\t', ' '), sep='\t', file=fout)

        # break

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Run MiniGPT model on a dataset.")
    parser.add_argument('--config', type=str, required=True, help='Input file path')
    args = parser.parse_args()
    import json
    configs=json.loads(open(args.config).read())
    args = argparse.Namespace(**configs)
    print (args)
    main(args)