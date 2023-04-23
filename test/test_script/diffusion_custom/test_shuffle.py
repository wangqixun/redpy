import torch
import os
import cv2
from tqdm import tqdm
from PIL import Image
from io import BytesIO
import numpy as np

from diffusers import ControlNetModel
from diffusers import DPMSolverMultistepScheduler
from diffusers.models import AutoencoderKL

from redpy.diffusers_custom import StableDiffusionCommonPipeline
from redpy.utils_redpy import get_basename_without_suffix

from controlnet_aux import ContentShuffleDetector


def face_shuffle():
    from PIL import Image
    from redpy.diffusers_custom import add_textual_inversion, add_lora, draw_kps
    generator = torch.manual_seed(0)
    generator_2 = torch.manual_seed(1)
    from redpy.grpc import CommonClient

    # 配置文件
    output_dir = '/share/wangqixun/workspace/tmp'
    base_model_path = '/share/xingchuan/code/diffusion/improved_sd/sd_diffusers/workspace/sam/sam_models'
    controlnet_path_list = [
        '/share/wangqixun/workspace/github_project/diffusers/checkpoint/controlnet/1-1/control_v11e_sd15_shuffle',
        '/share/wangqixun/workspace/github_project/diffusers/checkpoint/sam-control-face',
        '/share/wangqixun/workspace/github_project/model_dl/sam-control_v11p_sd15_canny',
    ]
    lora_path_list = [
        # '/share/wangqixun/workspace/github_project/diffusers/checkpoint/beautifulSky_beautifulSky.safetensors',
        '/share/wangqixun/workspace/github_project/diffusers/checkpoint/beautifuleyeslikeness_halfBody.safetensors',
    ]

    # face client
    client_face = CommonClient(
        ip='10.4.200.42',
        port='30390',
        max_send_message_length=100, 
        max_receive_message_length=100,
    )
    # depth client
    client_depth = CommonClient(
        ip='10.4.200.42',
        port='30301',
        max_send_message_length=100, 
        max_receive_message_length=100,
    )
    # shuffle
    processor = ContentShuffleDetector()

    # pipeline部分
    pipe = StableDiffusionCommonPipeline.from_pretrained(
        base_model_path,
        controlnet_list=controlnet_path_list,
        safety_checker=None,
        feature_extractor=None,
    )
    pipe = add_lora(lora_path_list, pipe, 0.4)
    pipe = pipe.cuda_half()


    # 用户图像
    img_file = '/share/axe/xhs_selfie_hd/00000e4c-f389-3054-baaf-8003e61f4d5c.jpg'
    image_raw = Image.open(img_file).convert('RGB')
    w, h = image_raw.size
    ratio = 512 / max(h, w)
    image_raw = image_raw.resize([round(ratio*w), round(ratio*h)])

    # 输出多张
    idx_start = 12
    for idx_image in range(idx_start, idx_start+6, 1):
        # infer
        prompt = "sam yang, 1girl, (yellow hair:1.1), (blue eyes:1.1), earring, 3D, (depth of field:1.3), side lighting, thin eyebrow, backlighting, (blurry background), (blush:0.9), (freckles:0.0), earrings, forehead, jewelry, pointy nose, red lips, shadow, solo, masterpiece, (looking at viewer:1.2)"
        negative_prompt = f"(cross-eyed:1.2), nsfw, painting by bad-artist-anime, painting by bad-artist, watermark, text, error, blurry, jpeg artifacts, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, artist name, worst quality, low quality, bad anatomy"
        # negative_prompt = f"({textual_inversion_tokens[1]}:0.8) (cross-eyed:1.2), nsfw, painting by bad-artist-anime, painting by bad-artist, watermark, text, error, blurry, jpeg artifacts, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, artist name, worst quality, low quality, bad anatomy"

        # control info - face
        control_image_bgr = np.array(image_raw)[..., ::-1]
        face_info = client_face.run([control_image_bgr])[0]
        face_emb = face_info['embedding']
        face_kps = face_info['kps']
        control_image_face = draw_kps(image_raw, face_kps)

        # control info - canny
        control_image_canny = np.array(image_raw)
        low_threshold = 50
        high_threshold = 200
        control_image_canny = cv2.Canny(control_image_canny, low_threshold, high_threshold)
        control_image_canny = np.concatenate([control_image_canny[..., None]] * 3, axis=2)
        control_image_canny = Image.fromarray(control_image_canny)

        # control info - canny
        depth_img = client_depth.run([np.array(image_raw)[..., ::-1]])
        depth_img = np.concatenate([depth_img[..., None]] * 3, axis=2)
        depth_img = Image.fromarray(depth_img)

        # shuffle
        control_image_shuffle = processor(image_raw)
        controlnet_conditioning = [
            dict(
                control_image=control_image_shuffle,
                control_index=0,
                control_weight=1.0,
            ),
        ]
        # image = pipe.img2img(
        #     image=image_raw,
        #     strength=1.0,
        #     controlnet_conditioning=controlnet_conditioning,
        #     prompt=prompt,
        #     negative_prompt=negative_prompt,
        #     guidance_scale=7.5,
        #     num_inference_steps=30,
        #     generator=generator,
        # ).images[0]
        image = pipe.text2img(
            controlnet_conditioning=controlnet_conditioning,
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=7.5,
            num_inference_steps=50,
            generator=generator,
        ).images[0]


        # control info - face
        control_image_bgr = np.array(image)[..., ::-1]
        face_info = client_face.run([control_image_bgr])[0]
        face_emb = face_emb
        face_kps = face_info['kps']
        control_image_face = draw_kps(image, face_kps)

        controlnet_conditioning = [
            dict(
                control_image=control_image_face,
                control_index=1,
                control_weight=1.0,
                control_visual_emb=face_emb,
            ),
            # dict(
            #     control_image=control_image_canny,
            #     control_index=1,
            #     control_weight=0.8,
            # ),
        ]

        # image = pipe.text2img(
        #     controlnet_conditioning=controlnet_conditioning,
        #     prompt=prompt,
        #     negative_prompt=negative_prompt,
        #     guidance_scale=7.5,
        #     # height=512,
        #     # width=512,
        #     num_inference_steps=50,
        #     generator=generator,
        # ).images[0]

        image = pipe.img2img(
            image=image,
            strength=0.8,
            controlnet_conditioning=controlnet_conditioning,
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=7.5,
            num_inference_steps=50,
            generator=generator_2,
        ).images[0]




        raw_image = Image.open(img_file).convert('RGB').resize((image.width, image.height))
        image = Image.fromarray(np.concatenate([np.array(image), np.array(raw_image), np.array(control_image_face)], axis=0))
        image.save(os.path.join(output_dir, f"{idx_image:04d}.jpg"))


def face_shuffle2():
    from PIL import Image
    from redpy.diffusers_custom import add_textual_inversion, add_lora, draw_kps
    generator = torch.manual_seed(0)
    generator_2 = torch.manual_seed(1)
    from redpy.grpc import CommonClient

    # 配置文件
    output_dir = '/share/wangqixun/workspace/tmp'
    base_model_path = '/share/xingchuan/code/diffusion/improved_sd/sd_diffusers/workspace/sam/sam_models'
    controlnet_path_list = [
        '/share/wangqixun/workspace/github_project/diffusers/checkpoint/controlnet/1-1/control_v11e_sd15_shuffle',
        '/share/wangqixun/workspace/github_project/diffusers/checkpoint/sam-control-face',
        '/share/wangqixun/workspace/github_project/model_dl/sam-control_v11p_sd15_canny',
    ]
    lora_path_list = [
        # '/share/wangqixun/workspace/github_project/diffusers/checkpoint/beautifulSky_beautifulSky.safetensors',
        '/share/wangqixun/workspace/github_project/diffusers/checkpoint/beautifuleyeslikeness_halfBody.safetensors',
    ]

    # face client
    client_face = CommonClient(
        ip='10.4.200.42',
        port='30390',
        max_send_message_length=100, 
        max_receive_message_length=100,
    )
    # depth client
    client_depth = CommonClient(
        ip='10.4.200.42',
        port='30301',
        max_send_message_length=100, 
        max_receive_message_length=100,
    )
    # shuffle
    processor = ContentShuffleDetector()

    # pipeline部分
    pipe = StableDiffusionCommonPipeline.from_pretrained(
        base_model_path,
        controlnet_list=controlnet_path_list,
        safety_checker=None,
        feature_extractor=None,
    )
    pipe = add_lora(lora_path_list, pipe, 0.4)
    pipe = pipe.cuda_half()


    # 用户图像
    img_file = '/share/axe/xhs_selfie_hd/00000e4c-f389-3054-baaf-8003e61f4d5c.jpg'
    image_raw = Image.open(img_file).convert('RGB')
    w, h = image_raw.size
    ratio = 512 / max(h, w)
    image_raw = image_raw.resize([round(ratio*w), round(ratio*h)])

    # 输出多张
    idx_start = 0
    for idx_image in range(idx_start, idx_start+6, 1):
        # infer
        prompt = "sam yang, 1girl, (yellow hair:1.1), (blue eyes:1.1), earring, 3D, (depth of field:1.3), side lighting, thin eyebrow, backlighting, (blurry background), (blush:0.9), (freckles:0.0), earrings, forehead, jewelry, pointy nose, red lips, shadow, solo, masterpiece, (looking at viewer:1.2)"
        negative_prompt = f"(cross-eyed:1.2), nsfw, painting by bad-artist-anime, painting by bad-artist, watermark, text, error, blurry, jpeg artifacts, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, artist name, worst quality, low quality, bad anatomy"
        # negative_prompt = f"({textual_inversion_tokens[1]}:0.8) (cross-eyed:1.2), nsfw, painting by bad-artist-anime, painting by bad-artist, watermark, text, error, blurry, jpeg artifacts, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, artist name, worst quality, low quality, bad anatomy"

        # control info - face
        control_image_bgr = np.array(image_raw)[..., ::-1]
        face_info = client_face.run([control_image_bgr])[0]
        face_emb = face_info['embedding']
        face_kps = face_info['kps']
        control_image_face = draw_kps(image_raw, face_kps)

        # control info - canny
        control_image_canny = np.array(image_raw)
        low_threshold = 50
        high_threshold = 200
        control_image_canny = cv2.Canny(control_image_canny, low_threshold, high_threshold)
        control_image_canny = np.concatenate([control_image_canny[..., None]] * 3, axis=2)
        control_image_canny = Image.fromarray(control_image_canny)

        # control info - canny
        depth_img = client_depth.run([np.array(image_raw)[..., ::-1]])
        depth_img = np.concatenate([depth_img[..., None]] * 3, axis=2)
        depth_img = Image.fromarray(depth_img)

        # shuffle
        control_image_shuffle = processor(image_raw)
        controlnet_conditioning = [
            dict(
                control_image=control_image_shuffle,
                control_index=0,
                control_weight=1.0,
            ),
            dict(
                control_image=control_image_face,
                control_index=1,
                control_weight=1.0,
                control_visual_emb=face_emb,
            ),
            # dict(
            #     control_image=control_image_canny,
            #     control_index=2,
            #     control_weight=0.8,
            # ),
        ]
        image = pipe.img2img(
            image=image_raw,
            strength=0.8,
            controlnet_conditioning=controlnet_conditioning,
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=7.5,
            num_inference_steps=50,
            generator=generator,
        ).images[0]




        raw_image = Image.open(img_file).convert('RGB').resize((image.width, image.height))
        image = Image.fromarray(np.concatenate([np.array(image), np.array(raw_image), np.array(control_image_face)], axis=0))
        image.save(os.path.join(output_dir, f"{idx_image:04d}.jpg"))


if __name__ == '__main__':
    # func_test()
    # func_test_face()
    # face_shuffle()
    face_shuffle2()