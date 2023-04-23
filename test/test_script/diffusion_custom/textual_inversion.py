
import torch
import os
import cv2
from tqdm import tqdm
from PIL import Image
from io import BytesIO
from diffusers.models import AutoencoderKL
import numpy as np
import json


from redpy.grpc import CommonClient
from redpy.diffusers_custom import StableDiffusionCommonPipeline
from redpy.utils_redpy import get_basename_without_suffix, load_json
from redpy.diffusers_custom import add_lora_unet_and_text_encoder, add_lora_bin, add_textual_inversion


def f1():
    generator = torch.manual_seed(0)
    output_dir = '/share/wangqixun/workspace/tmp/tmp_textual_inversion'

    base_model = '/share/wangqixun/workspace/github_project/diffusers/checkpoint/Chilloutmix-Ni'
    textual_inversion_file = [
        # '/share/wangqixun/workspace/github_project/diffusers/checkpoint/EasyNegative.pt'
        # '/share/wangqixun/workspace/github_project/diffusers/checkpoint/Style-Neeko.pt',
        # '/share/wangqixun/workspace/github_project/diffusers/checkpoint/bad-picture-chill-75v.pt'
        # '/share/wangqixun/workspace/github_project/diffusers/checkpoint/style-miaozu-20000.pt',
        '/share/wangqixun/workspace/github_project/diffusers/checkpoint/Style-Neeko.pt'
    ]
    pipe = StableDiffusionCommonPipeline.from_pretrained(
        base_model,
        controlnet_list=[],
        safety_checker=None,
        feature_extractor=None,
    )
    pipe, textual_inversion_tokens = add_textual_inversion(pipe, textual_inversion_file)
    pipe = pipe.cuda_half()

    os.makedirs(output_dir, exist_ok=True)

    start_idx = 18
    for idx_img, img_name in enumerate(range(6)):
        img_name = str(img_name)
        basename_without_suffix = get_basename_without_suffix(img_name)

        prompt = f"({textual_inversion_tokens[0]}:1.0) ((cactus with eyes)) (beautiful eyes), cute face, photo realistic, 20 megapixel, nikon d850, ((full body intricate, vibrant, photo realistic, realistic, dramatic, sharp focus, 8k)), subsurface scattering, sharp, retouched, intricate detail,, highly detailed, digital painting, artstation, concept art, smooth, sharp focus, illustration, art by krenz cushart and artgerm demura and raja ravi verma"
        # negative_prompt = f" many fingers, "
        # negative_prompt = f"({textual_inversion_tokens[0]}:0.6), many fingers, "
        negative_prompt = f"asian, cartoon, 3d, (disfigured), (bad art), (deformed), bad hand, extra fingers, (poorly drawn), (extra limbs), strange colours, blurry, boring, sketch, lackluster, big breast, large breast, huge breasts, self-portrait, signature, letters, watermark, desaturated, monochrome"
        controlnet_conditioning = []

        # img2img
        # image = pipe.img2img(
        #     image=input_image,
        #     controlnet_conditioning=controlnet_conditioning,
        #     strength=0.5,
        #     prompt=prompt,
        #     negative_prompt=negative_prompt,
        #     guidance_scale=5.,
        #     num_inference_steps=30,
        # ).images[0]

        # text2img
        image = pipe.text2img(
            controlnet_conditioning=controlnet_conditioning,
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=7.5,
            height=768,
            width=512,
            num_inference_steps=50,
            generator=generator,
        ).images[0]

        image.save(f"{output_dir}/{start_idx+idx_img:04d}_{basename_without_suffix}.jpg")




if __name__ == "__main__":
    f1()




