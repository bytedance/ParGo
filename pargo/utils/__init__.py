# Copyright (c) 2024 Bytedance Ltd.
# SPDX-License-Identifier: BSD-3-Clause
"""Utilities."""

import os
from PIL import Image
from torch.hub import download_url_to_file, urlparse, get_dir

# helper functions
def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def is_url(url_or_filename):
    parsed = urlparse(url_or_filename)
    return parsed.scheme in ("http", "https")

def is_hdfs(path):
    return path.startswith('hdfs')

def is_dir(path:str):
    return (path.endswith('/')) or ('.' not in os.path.basename(path))


def import_from(module, name):
    module = __import__(module, fromlist=[name])
    return getattr(module, name)


def get_cache_dir(child_dir=''):
    hub_dir = get_dir()
    child_dir = () if not child_dir else (child_dir,)
    model_dir = os.path.join(hub_dir, 'checkpoints', *child_dir)
    os.makedirs(model_dir, exist_ok=True)
    return model_dir


def remove_exif(image_name):
    image = Image.open(image_name)
    if not image.getexif():
        return
    data = list(image.getdata())
    image_without_exif = Image.new(image.mode, image.size)
    image_without_exif.putdata(data)
    return image_without_exif
