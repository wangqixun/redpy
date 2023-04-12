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
from redpy.grpc import DepthEstimationClient
from redpy.utils_redpy import get_basename_without_suffix


def func_test():
    output_dir = '/share/wangqixun/workspace/tmp'
    base_model_path = '/share/xingchuan/code/diffusion/improved_sd/sd_diffusers/workspace/sam/sam_models'
    controlnet_path_list = [
        '/share/wangqixun/workspace/github_project/model_dl/sam_models-control-canny',
        '/share/wangqixun/workspace/github_project/model_dl/sam-control-depth',
    ]

    os.makedirs(output_dir, exist_ok=True)

    pipe = StableDiffusionCommonPipeline.from_pretrained(
        base_model_path,
        controlnet_list=controlnet_path_list,
        safety_checker=None,
        torch_dtype=torch.float16,
        feature_extractor=None,
    )
    pipe = pipe.to(torch.float16)
    pipe = pipe.to('cuda:0')

    img_input_dir = '/share/wangqixun/data/test_data/female_face'
    img_name_list = sorted(os.listdir(img_input_dir))

    client = DepthEstimationClient(
        ip='10.4.200.43',
        port='31002',
        max_send_message_length=100, 
        max_receive_message_length=100,
    )


    start_idx = 0
    for idx_img, img_name in enumerate(tqdm(img_name_list)):
        basename_without_suffix = get_basename_without_suffix(img_name)
        img_file = os.path.join(img_input_dir, img_name)
        input_image = Image.open(img_file).convert('RGB')
        w, h = input_image.size
        ratio = 512 / min(h, w)
        input_image = input_image.resize([int(ratio*w), int(ratio*h)])
        w, h = input_image.size
        ratio = 768 / max(h, w)
        input_image = input_image.resize([int(ratio*w), int(ratio*h)])
        control_image = np.array(input_image)
        low_threshold = 50
        high_threshold = 200
        control_image = cv2.Canny(control_image, low_threshold, high_threshold)
        control_image = np.concatenate([control_image[..., None]] * 3, axis=2)
        control_image = Image.fromarray(control_image)

        depth_img = client.run(np.array(input_image)[..., ::-1])
        depth_img = np.concatenate([depth_img[..., None]] * 3, axis=2)
        depth_img = Image.fromarray(depth_img)


        prompt = "sam yang, 1girl, backlighting, bare shoulders, black choker, blurry, blurry background, blush, breasts, choker, cleavage, closed mouth, collarbone, earrings, forehead, freckles, hair over shoulder, jewelry, long hair, looking down, pointy nose, red lips, shadow, solo, thick eyebrows, thick eyelashes, upper body, white hair , ((masterpiece)) <lora:sam_yang_offset:1>"
        negative_prompt = "(painting by bad-artist-anime:0.9), (painting by bad-artist:0.9), watermark, text, error, blurry, jpeg artifacts, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, artist name, (worst quality, low quality:1.4), bad anatomy"

        controlnet_conditioning = [
            dict(
                control_image=depth_img,
                control_index=1,
                control_weight=1.0,
            ),
            dict(
                control_image=control_image,
                control_index=0,
                control_weight=0.5,
            ),
        ]
        # controlnet_conditioning = []

        # img2img
        # image = pipe.img2img(
        #     image=input_image,
        #     controlnet_conditioning=controlnet_conditioning,
        #     strength=0.8,
        #     prompt=prompt,
        #     negative_prompt=negative_prompt,
        #     num_inference_steps=50,
        # ).images[0]
        # image_np = np.array(image)
        # input_image_np = np.array(input_image)
        # if image_np.shape != input_image_np.shape:
        #     image_np = cv2.resize(image_np, (input_image_np.shape[1], input_image_np.shape[0]))
        # image_np = np.concatenate([image_np, input_image_np], axis=1)
        # image = Image.fromarray(image_np)

        # text2img
        # image = pipe.text2img(
        #     controlnet_conditioning=controlnet_conditioning,
        #     prompt=prompt,
        #     negative_prompt=negative_prompt,
        #     height=768,
        #     width=512,
        #     num_inference_steps=50,
        # ).images[0]

        # inpatinting
        mask_image = np.zeros_like(input_image)
        mask_image[100:400, ] = 255
        mask_image = Image.fromarray(mask_image)
        image = pipe.inpatinting(
            image=input_image,
            mask_image=mask_image,
            controlnet_conditioning=controlnet_conditioning,
            prompt=prompt,
            negative_prompt=negative_prompt,
            # height=768,
            # width=512,
            num_inference_steps=50,
        ).images[0]
        image_np = np.array(image)
        input_image_np = np.array(input_image)
        if image_np.shape != input_image_np.shape:
            image_np = cv2.resize(image_np, (input_image_np.shape[1], input_image_np.shape[0]))
        image_np = np.concatenate([image_np, input_image_np], axis=1)
        image = Image.fromarray(image_np)

        image.save(f"{output_dir}/{basename_without_suffix}_{start_idx+idx_img:04d}.jpg")


def func_test_face():
    from PIL import Image
    from redpy.diffusers_custom import add_textual_inversion, add_lora, draw_kps
    generator = torch.manual_seed(999)
    from redpy.grpc import CommonClient
    from redpy.grpc import DepthEstimationClient

    output_save_dir = '/share/wangqixun/workspace/tmp/face_control_tmp'
    img_file = '/share/axe/xhs_selfie_hd/00000e4c-f389-3054-baaf-8003e61f4d5c.jpg'

    pretrained_model_name_or_path = '/share/wangqixun/workspace/github_project/diffusers/checkpoint/stable-diffusion-v1-5'
    controlnet_path_list = [
        '/share/wangqixun/workspace/github_project/diffusers/checkpoint/controlnet/exp/face_conrtol_v2/ControlNetModel',
        # '/share/wangqixun/workspace/github_project/diffusers/checkpoint/sam-control-face',
        '/share/wangqixun/workspace/github_project/model_dl/sam_models-control-canny',
        # '/share/wangqixun/workspace/github_project/model_dl/sam-control-depth',
        # '/share/wangqixun/workspace/github_project/diffusers/checkpoint/controlnet/sd-controlnet-face',
        # '/share/wangqixun/workspace/github_project/diffusers/checkpoint/sd-controlnet-canny'
    ]
    textual_inversion_file = [
        '/share/wangqixun/workspace/github_project/diffusers/checkpoint/bad-picture-chill-75v.pt'
    ]
    lora_path_list = [
        '/share/wangqixun/workspace/github_project/diffusers/checkpoint/beautifulSky_beautifulSky.safetensors',
        '/share/wangqixun/workspace/github_project/diffusers/checkpoint/beautifuleyeslikeness_halfBody.safetensors',
    ]

    controlnet_list = controlnet_path_list

    pipe = StableDiffusionCommonPipeline.from_pretrained(
        pretrained_model_name_or_path,
        controlnet_list=controlnet_list,
        safety_checker=None,
        feature_extractor=None,
    )
    pipe = pipe.to('cuda:0')
    pipe = add_lora(lora_path_list, pipe, 0.0)
    pipe = pipe.to(torch.float16)
    pipe.vae = pipe.vae.to(torch.float32)


    image_raw = Image.open(img_file).convert('RGB')

    w, h = image_raw.size
    ratio = 448 / min(h, w)
    image_raw = image_raw.resize([int(ratio*w), int(ratio*h)])
    w, h = image_raw.size
    ratio = 512 / max(h, w)
    image_raw = image_raw.resize([int(ratio*w), int(ratio*h)])
    client = CommonClient(
        ip='10.4.200.40',
        port='30390',
        max_send_message_length=100, 
        max_receive_message_length=100,
    )
    client_depth = DepthEstimationClient(
        ip='10.4.200.43',
        port='31002',
        max_send_message_length=100, 
        max_receive_message_length=100,
    )

    control_image_bgr = np.array(image_raw)[..., ::-1]
    face_info = client.run([control_image_bgr])[0]
    face_emb = face_info['embedding']
    face_kps = face_info['kps']
    control_image_face = draw_kps(image_raw, face_kps)

    control_image_canny = np.array(image_raw)
    low_threshold = 50
    high_threshold = 200
    control_image_canny = cv2.Canny(control_image_canny, low_threshold, high_threshold)
    control_image_canny = np.concatenate([control_image_canny[..., None]] * 3, axis=2)
    control_image_canny = Image.fromarray(control_image_canny)

    idx_start = 36
    for iii in range(idx_start, idx_start+6, 1):
        prompt = f'van gogh style, created by van gogh, portrait of 25 year old girl with (white long hair:1.1), looking at viewer, sky, sea'
        negative_prompt = 'canvas frame, cartoon, 3d, ((disfigured)), ((bad art)), ((deformed)),((extra limbs)),((close up)),'
        # prompt = "sam yang, 1girl, 3D, (depth of field:1.3), side lighting, thin eyebrow, (brown eyes:1.2), backlighting, (blurry background), (blush:0.9), (freckles:0.0), earrings, forehead, jewelry, pointy nose, red lips, shadow, solo, masterpiece, (looking at viewer:1.2)"
        # negative_prompt = "(cross-eyed:1.2), nsfw, painting by bad-artist-anime, painting by bad-artist, watermark, text, error, blurry, jpeg artifacts, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, artist name, worst quality, low quality, bad anatomy"
        controlnet_conditioning = [
            dict(
                control_image=control_image_face,
                control_index=0,
                control_weight=1.,
                control_visual_emb=face_emb,
            ),
            # dict(
            #     control_image=control_image_canny,
            #     control_index=1,
            #     control_weight=0.75,
            # ),
            # dict(
            #     control_image=depth_img,
            #     control_index=2,
            #     control_weight=0.5,
            # ),
        ]
        # controlnet_conditioning = []

        image = pipe.text2img(
            controlnet_conditioning=controlnet_conditioning,
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=7.5,
            # height=512,
            # width=512,
            num_inference_steps=50,
            generator=generator,
        ).images[0]

        # image = pipe.img2img(
        #     image=image_raw,
        #     strength=0.85,
        #     controlnet_conditioning=controlnet_conditioning,
        #     prompt=prompt,
        #     negative_prompt=negative_prompt,
        #     guidance_scale=7.5,
        #     num_inference_steps=50,
        #     generator=generator,
        # ).images[0]


        raw_image = Image.open(img_file).convert('RGB').resize((image.width, image.height))
        image = Image.fromarray(np.concatenate([np.array(image), np.array(raw_image)], axis=0))
        image.save(os.path.join(output_save_dir, f"{iii:04d}.jpg"))




if __name__ == '__main__':
    # func_test()
    func_test_face()