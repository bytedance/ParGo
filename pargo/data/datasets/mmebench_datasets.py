# Copyright (c) 2024 Bytedance Ltd.
# SPDX-License-Identifier: BSD-3-Clause
import base64
import io
import random
from symbol import continue_stmt

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import os
def resize_and_pad(image, target_size, pad_color=(225, 225, 225)):
    ratio = min(target_size[0] / image.width, target_size[1] / image.height)
    new_size = (int(image.width * ratio), int(image.height * ratio))
    resized_image = image.resize(new_size, Image.BICUBIC)
    padded_image = Image.new("RGB", target_size, pad_color)
    pad_left = (target_size[0] - new_size[0]) // 2
    pad_top = (target_size[1] - new_size[1]) // 2
    padded_image.paste(resized_image, (pad_left, pad_top))
    
    return padded_image

class MMEBenchDataset(Dataset):
    def __init__(self,
                 data_file,
                 sys_prompt='There are several options:',
                 prompt_style='minigpt_v2',
                 root='/mnt/bd/nerson/benchmark/mme/eval_tool',
                 image_size=224,
                 prompt_template="<Img><ImageHere></Img> {}",
                 keep_pad=False,
                 keep_pad_large=True):
        self.root = root
        
        self.data = self.read_file(data_file)
        mean = (0.48145466, 0.4578275, 0.40821073)
        std = (0.26862954, 0.26130258, 0.27577711)
        self._other_configs = {
            "image_size": 336,
            "mean": mean,
            "std": std,
            "min_scale": 1.0,
            "max_scale":  1.0,
        }
        
        self.prompt_template = prompt_template
        print ("self.prompt_template", self.prompt_template)
        self.sys_prompt =""
        if keep_pad_large:
            self.vis_processor = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ])
        else:
            self.vis_processor = transforms.Compose([
                transforms.Resize((image_size, image_size), interpolation=transforms.functional.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ])
        self.keep_pad = keep_pad
        self.keep_pad_large = keep_pad_large
        self.resize_image = 448

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        image_path = self.data[idx]['image']
        
        image = Image.open(image_path)
        if image.format == "PNG":
            try:
                background = Image.new('RGB', image.size, (255, 255, 255))
                background.paste(image, mask=image.split()[3])
                image = background.copy()
            except:
                image = image.convert('RGB')
        else:
            image = image.convert('RGB')

        if self.keep_pad:
            image = expand2square(image, tuple(int(x*255) for x in (0.48145466, 0.4578275, 0.40821073)))
        elif self.keep_pad_large:
            image = resize_and_pad(image,(self.resize_image,self.resize_image),pad_color=tuple(int(x*255) for x in self._other_configs["mean"]))
        image = self.vis_processor(image).unsqueeze(0) 
        question = self.data[idx]['question']
        answer = self.data[idx]['answer'] 
        prompt = question
        data = {
            'img': image.squeeze(),
            'answer': answer,
            'prompts':self.prompt_template.format(prompt),
            'img_path':image_path.split('/')[-1]
        }
        return data
    def read_file(self,file):
        evaluations=[]
        
        with open(file, 'r', encoding='utf-8') as fin:
            lines = fin.read().splitlines()
            for line in lines:
                img, question, gt = line.strip().split('\t')
                img_path = os.path.join(self.root,'images',file.split('/')[-1].replace('.txt',''), img)
                
                try:
                    assert os.path.exists(img_path), img_path
                except Exception as e:
                    print ('skip img {}'.format(img_path))
                    continue
                
                evaluations.append({'image':img_path,'question':question, 'answer':gt})
        return evaluations

def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result