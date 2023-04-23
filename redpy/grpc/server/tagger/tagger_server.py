import cv2
import torch
import numpy as np
from PIL import Image
import queue

from huggingface_hub import from_pretrained_keras
import base64
import json
import os

import pandas as pd
from flask import Flask, jsonify, request


from redpy.grpc.server.common.server import convert_to_server


def smart_imread(img, flag=cv2.IMREAD_UNCHANGED):
    if img.endswith(".gif"):
        img = Image.open(img)
        img = img.convert("RGB")
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    else:
        img = cv2.imread(img, flag)
    return img


def smart_24bit(img):
    if img.dtype is np.dtype(np.uint16):
        img = (img / 257).astype(np.uint8)

    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        trans_mask = img[:, :, 3] == 0
        img[trans_mask] = [255, 255, 255, 255]
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def make_square(img, target_size):
    old_size = img.shape[:2]
    desired_size = max(old_size)
    desired_size = max(desired_size, target_size)

    delta_w = desired_size - old_size[1]
    delta_h = desired_size - old_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [255, 255, 255]
    new_im = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    return new_im


def smart_resize(img, size):
    # Assumes the image has already gone through make_square
    if img.shape[0] > size:
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    elif img.shape[0] < size:
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)
    return img


queue_maxsize = 2


class Tagger(object):
    
    def __init__(self, queue_maxsize=1, device_list=["cuda:0"], dim=448):
        self.queue = queue.Queue(maxsize=queue_maxsize)
        self.device_list = device_list
        self.dim = dim
        self.thresh = 0.3537
        self.label_names = pd.read_csv("/root/SW-CV-ModelZoo/selected_tags.csv")

        for i in range(queue_maxsize):

            model = from_pretrained_keras("SmilingWolf/wd-v1-4-convnextv2-tagger-v2")

            # warm up
            image = np.zeros([dim, dim, 3], dtype=np.uint8)
            img = image
            img = smart_24bit(img)
            img = make_square(img, dim)
            img = smart_resize(img, dim)
            img = img.astype(np.float32)
            img = np.expand_dims(img, 0)
            probs = model.predict(img)

            self.queue.put([model])        

    
    @convert_to_server("TaggerServer", 30126, queue_maxsize)
    def infer(self, img):
        model = self.queue.get(True, timeout=10)[0]

        img = smart_24bit(img)
        img = make_square(img, self.dim)
        img = smart_resize(img, self.dim)
        img = img.astype(np.float32)
        img = np.expand_dims(img, 0)
        probs = model.predict(img)

        found_tags = self.label_names[probs[0] > self.thresh][["name", "category"]]

        labels_list = []
        for index, pair in found_tags.iterrows():
            if pair["category"] == 0:
                labels_list.append(' '.join(pair["name"].split('_')))

        if model is not None:
            self.queue.put([model])        

        return labels_list


if __name__ == '__main__':
    
    # load model
    tagger = Tagger(queue_maxsize=queue_maxsize)
    
    # load image
    # image = np.zeros([512, 512, 3], dtype=np.uint8) + 128
    image = cv2.imread('/share/wangqixun/workspace/github_project/multimodal-intelligence/wqx_business/pixar/data/person/2.jpg')
    
    # infer
    tags = tagger.infer(image)
    print(tags)