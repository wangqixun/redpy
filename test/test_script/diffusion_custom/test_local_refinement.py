from annotator.openpose import OpenposeDetector

import torch
import os
import cv2
from tqdm import tqdm
from PIL import Image
from io import BytesIO
from diffusers.models import AutoencoderKL
import numpy as np
import json

from redpy.grpc import DepthEstimationClient, CommonClient
from redpy.diffusers_custom import StableDiffusionCommonPipeline
from redpy.utils_redpy import get_basename_without_suffix, load_json
from redpy.diffusers_custom import add_lora_unet_and_text_encoder, add_lora_bin, add_textual_inversion
from redpy.diffusers_custom import add_lora_bin, draw_kps



if __name__ == '__main__':

    img_input_dir = '/share/wangqixun/data/test_data/景大人小测试集'
    output_dir = '/share/wangqixun/workspace/tmp/tmp_redmatcha'

    base_model = "/share/wangqixun/workspace/github_project/diffusers/checkpoint/fantasyWorld_v1"
    lora_model = '/share/wangqixun/workspace/github_project/diffusers/checkpoint/Lora/lora_redmatcha_v1_2_1500/pytorch_lora_weights.bin'
    textual_inversion_file = [
        '/share/wangqixun/workspace/github_project/diffusers/checkpoint/bad-picture-chill-75v.pt'
    ]
    controlnet_path_list = [
        '/share/wangqixun/workspace/github_project/diffusers/checkpoint/controlnet/exp/face_conrtol_v2/ControlNetModel',
        '/share/wangqixun/workspace/github_project/diffusers/checkpoint/sd-controlnet-openpose',
    ]
    
    # vae_path = '/share/wangqixun/workspace/github_project/diffusers/checkpoint/anything-v4.0.vae'

    pipe = StableDiffusionCommonPipeline.from_pretrained(
        base_model,
        controlnet_list=controlnet_path_list,
        safety_checker=None,
        torch_dtype=torch.float16,
        feature_extractor=None,
    )
    # pipe.unet.load_attn_procs(lora_model)
    pipe = add_lora_bin(lora_model, pipe, 1.0)
    pipe, textual_inversion_tokens = add_textual_inversion(pipe, textual_inversion_file)
    pipe = pipe.to(torch.float32)
    # pipe.vae = pipe.vae.to(torch.float32)
    pipe = pipe.to('cuda:0')
    img_name_list = sorted(os.listdir(img_input_dir))
    os.makedirs(output_dir, exist_ok=True)

    # client = DepthEstimationClient(
    #     ip='10.4.200.43',
    #     port='31002',
    #     max_send_message_length=100, 
    #     max_receive_message_length=100,
    # )
    openpose = OpenposeDetector()
    client = CommonClient(
        ip='10.4.200.40',
        port='30390',
        max_send_message_length=100, 
        max_receive_message_length=100,
    )


    start_idx = 0
    for idx_img, img_name in enumerate(tqdm(img_name_list)):
        basename_without_suffix = get_basename_without_suffix(img_name)
        if "IMG_5985" not in basename_without_suffix:
            continue

        img_file = os.path.join(img_input_dir, img_name)
        input_image = Image.open(img_file).convert('RGB')
        w, h = input_image.size
        ratio = 512 / min(h, w)
        input_image = input_image.resize([int(ratio*w), round(ratio*h)])
        w, h = input_image.size
        ratio = 512 / max(h, w)
        input_image = input_image.resize([int(ratio*w), round(ratio*h)])
        input_image.save(f"/share/wangqixun/workspace/tmp/0000.jpg")

        # openpose
        detected_map = np.array(input_image)
        detected_map, _ = openpose(detected_map)
        detected_map = Image.fromarray(detected_map)

        controlnet_conditioning = [
            dict(
                control_image=detected_map,
                control_index=1,
                control_weight=1.0,
            ),
        ]

        prompt = f"<redmatcha>, "
        negative_prompt = f"({textual_inversion_tokens[0]}:0.6), many fingers, "
        image = pipe.img2img(
            image=input_image,
            strength=0.6,
            controlnet_conditioning=controlnet_conditioning,
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=5.,
            num_inference_steps=30,
            bbox=None,
        ).images[0]
        image.save(f"/share/wangqixun/workspace/tmp/0001.jpg")

        # face
        control_image_bgr = np.array(input_image)[..., ::-1]
        face_info = client.run([control_image_bgr])

        



