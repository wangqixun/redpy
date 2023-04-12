from safetensors import safe_open
from diffusers import StableDiffusionPipeline
import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.models import AutoencoderKL
import os


def mix_pipeline_safetensors():
    pipeline_1_path = "/share/wangqixun/workspace/github_project/diffusers/checkpoint/Chilloutmix-Ni"
    pipeline_2_path = "/share/wangqixun/workspace/github_project/diffusers/checkpoint/uberRealisticPornMerge_urpmv12"



    #Return a CheckpointMergerPipeline class that allows you to merge checkpoints. 
    #The checkpoint passed here is ignored. But still pass one of the checkpoints you plan to 
    #merge for convenience
    # pipe = DiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", custom_pipeline="checkpoint_merger")

    #There are multiple possible scenarios:
    #The pipeline with the merged checkpoints is returned in all the scenarios

    #Compatible checkpoints a.k.a matched model_index.json files. Ignores the meta attributes in model_index.json during comparision.( attrs with _ as prefix )
    # merged_pipe = pipe.merge(["CompVis/stable-diffusion-v1-4","CompVis/stable-diffusion-v1-2"], interp = "sigmoid", alpha = 0.4)

    #Incompatible checkpoints in model_index.json but merge might be possible. Use force = True to ignore model_index.json compatibility
    # merged_pipe_1 = pipe.merge(["CompVis/stable-diffusion-v1-4","hakurei/waifu-diffusion"], force = True, interp = "sigmoid", alpha = 0.4)

    #Three checkpoint merging. Only "add_difference" method actually works on all three checkpoints. Using any other options will ignore the 3rd checkpoint.
    # merged_pipe_2 = pipe.merge(["CompVis/stable-diffusion-v1-4","hakurei/waifu-diffusion","prompthero/openjourney"], force = True, interp = "add_difference", alpha = 0.4)

    # prompt = "An astronaut riding a horse on Mars"

    # image = merged_pipe(prompt).images[0]

    pipe = DiffusionPipeline.from_pretrained(pipeline_2_path, custom_pipeline="checkpoint_merger")
    merged_pipe = pipe.merge([pipeline_2_path, pipeline_1_path], interp="sigmoid", alpha=0.7)


def is_int(d):
    try:
        d = int(d)
        return True
    except Exception as e:
        return False


def add_lora(lora_path, pipe, lora_weight=0.5):
    tensors = {}
    with safe_open(lora_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key).to('cuda')


    for k_lora, v_lora in tensors.items():
        if k_lora.startswith('lora_te'):
            model = pipe.text_encoder
            # continue
        elif k_lora.startswith('lora_unet'):
            model = pipe.unet
        else:
            print(k_lora)
        
        # down 跳过
        if '.lora_down.' in k_lora:
            print('lora_down')
            continue
        if '.alpha' in k_lora:
            print('alpha')
            continue
        print('lora_up')

        k_lora_name = k_lora.split('.')[0]
        attr_name_list = k_lora_name.split('_')
        cur_attr = model
        latest_attr_name = ''
        for idx in range(2, len(attr_name_list)):
            attr_name = attr_name_list[idx]
            if is_int(attr_name):
                cur_attr = cur_attr[int(attr_name)]
                latest_attr_name = ''
            else:
                try:
                    if latest_attr_name != '':
                        cur_attr = cur_attr.__getattr__(f"{latest_attr_name}_{attr_name}")
                    else:
                        cur_attr = cur_attr.__getattr__(attr_name)
                    latest_attr_name = ''
                except Exception as e:
                    if latest_attr_name != '':
                        latest_attr_name = f"{latest_attr_name}_{attr_name}"
                    else:
                        latest_attr_name = attr_name

        w = cur_attr.weight
        up_w = v_lora
        down_w = tensors[k_lora.replace('.lora_up.', '.lora_down.')]
        print(down_w.shape, up_w.shape, w.shape)
        try:
            alpha_key = k_lora_name + '.alpha'
            alpha_w = tensors[alpha_key]
            wight = alpha_w / up_w.shape[1]
            print(alpha_w, wight)
        except Exception as e:
            wight = 1
        
        einsum_a = f"ijabcdefg"
        einsum_b = f"jkabcdefg"
        einsum_res = f"ikabcdefg"
        length_shape = len(up_w.shape)
        einsum_str = f"{einsum_a[:length_shape]},{einsum_b[:length_shape]}->{einsum_res[:length_shape]}"
        d_w = torch.einsum(einsum_str, up_w, down_w)
        # print(d_w.shape, wight)

        # wight = 1
        cur_attr.weight.data = cur_attr.weight.data + d_w * wight * lora_weight
        print('================================= add')
    return pipe




def test_model():

    output_dir = '/share/wangqixun/workspace/tmp/tmp_6'
    os.makedirs(output_dir, exist_ok=True)

    # model_id = "/share/wangqixun/workspace/github_project/diffusers/checkpoint/Chilloutmix-Ni"
    # model_id = "/share/wangqixun/workspace/github_project/diffusers/checkpoint/abyssorangemix2SFW_abyssorangemix2Sfw"
    # model_id = "/share/wangqixun/workspace/github_project/diffusers/checkpoint/Midnight_Mixer_Melt"
    # model_id = "/share/wangqixun/workspace/github_project/diffusers/checkpoint/URPM_CM_mix"
    # model_id = "/share/wangqixun/workspace/github_project/diffusers/checkpoint/uberRealisticPornMerge_urpmv12"
    model_id = "/share/wangqixun/workspace/github_project/diffusers/checkpoint/perfectWorld_perfectWorldBakedVAE"
    # model_id = "/share/wangqixun/workspace/github_project/diffusers/checkpoint/Midnight_Mixer_Melt"
    # model_id = "/share/wangqixun/workspace/github_project/diffusers/checkpoint/cheeseDaddys_35"
    # model_id = "/share/wangqixun/workspace/github_project/diffusers/checkpoint/abyssorangemix3AOM3_aom3a3"
    # model_id = "/share/wangqixun/workspace/github_project/diffusers/checkpoint/gf2_v20"
    # model_id = "/share/wangqixun/workspace/github_project/diffusers/checkpoint/neverendingDreamNED_bakedVae"
    # model_id = "/share/wangqixun/workspace/github_project/diffusers/checkpoint/Anything-v4.5-vae-fp16-diffuser"
    # model_id = "/share/wangqixun/workspace/github_project/diffusers/checkpoint/abyssorangemix2SFW_abyssorangemix2Sfw"
    # model_id = "/share/wangqixun/workspace/github_project/diffusers/checkpoint/wlop_1"

    

    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, 
        torch_dtype=torch.float32, 
        custom_pipeline="lpw_stable_diffusion",
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    # pipe.vae = AutoencoderKL.from_pretrained('/share/wangqixun/workspace/github_project/diffusers/checkpoint/stabilityai/sd-vae-ft-mse')
    pipe = pipe.to(torch.float16)
    pipe = pipe.to('cuda')
    pipe.safety_checker = None

    # lora_path = "/share/wangqixun/workspace/github_project/diffusers/checkpoint/LORADilraba_v10.safetensors"
    # pipe = add_lora(lora_path, pipe, 0.7)
    # lora_path = "/share/wangqixun/workspace/github_project/diffusers/checkpoint/galGadotLora_v10.safetensors"
    # pipe = add_lora(lora_path, pipe, 0.65)
    # lora_path = "/share/wangqixun/workspace/github_project/diffusers/checkpoint/taiwanDollLikeness_v10.safetensors"
    # pipe = add_lora(lora_path, pipe, 0.5)
    # lora_path = "/share/wangqixun/workspace/github_project/diffusers/checkpoint/koreanDollLikeness_v10.safetensors"
    # pipe = add_lora(lora_path, pipe, 0.5)
    # lora_path = "/share/wangqixun/workspace/github_project/diffusers/checkpoint/fashionGirl_v35.safetensors"
    # pipe = add_lora(lora_path, pipe, 0.66)
    # lora_path = "/share/wangqixun/workspace/github_project/diffusers/checkpoint/gakkiAragakiYui_v2.safetensors"
    # pipe = add_lora(lora_path, pipe, 0.65)
    # lora_path = "/share/wangqixun/workspace/github_project/diffusers/checkpoint/asiafacemixLora300_asiafacemixPruned.safetensors"
    # pipe = add_lora(lora_path, pipe, 0.5)
    # lora_path = "/share/wangqixun/workspace/github_project/diffusers/checkpoint/nudify_v11.safetensors"
    # pipe = add_lora(lora_path, pipe, 0.6)
    # lora_path = "/share/wangqixun/workspace/github_project/diffusers/checkpoint/hipoly3DModelLora_v10.safetensors"
    # pipe = add_lora(lora_path, pipe, 0.5)
    # lora_path = "/share/wangqixun/workspace/github_project/diffusers/checkpoint/makotoShinkaiSubstyles_offset.safetensors"
    # pipe = add_lora(lora_path, pipe, 1.0)
    lora_path = "/share/wangqixun/workspace/github_project/diffusers/checkpoint/jiyeon_V30.safetensors"
    pipe = add_lora(lora_path, pipe, 0.3)



    # hiqcgbody, hiqcgface
    text = 'jiyeon2, photorealistic,realistic, solo, photorealistic, best quality, ultra high res, round hat with ribbon, short curly blonde hair, big blue eyes, sitting in a field of wildflowers, beautiful, masterpiece, best quality, extremely detailed face, perfect lighting, close up photo, best quality, ultra high res, photorealistic, ultra detailed, masterpiece, best quality, (naked: 1.5),'
    neg_text = '(worst quality:2), (low quality:2), (normal quality:2), lowres, normal quality, ((monochrome)), ((grayscale)), skin spots, acnes, skin blemishes, age spot, glans,watermark,nsfw,'
    batch_size = 2
    input_size = (512, 640)
    latents = torch.randn([batch_size, 4, input_size[1] // 8, input_size[0] // 8], device=pipe.device, dtype=torch.float16)


    for idx_img in range(0, 9, 1):
        imgs = pipe(
            text, 
            latents=None,
            negative_prompt=neg_text, 
            height=input_size[1], 
            width=input_size[0],
            num_inference_steps=50,
            # guidance_scale=8,
            num_images_per_prompt=batch_size,
        ).images
        # imgs[0].save(f'/share/wangqixun/workspace/tmp/{idx_img:03d}.jpg')
        for idx in range(len(imgs)):
            imgs[idx].save(f'{output_dir}/{idx_img:03d}_{idx:03d}.jpg')


def merge_imgs_to_video():
    imgs_dir = '/share/wangqixun/workspace/tmp_5'
    output_video_file = '/share/wangqixun/workspace/tmp_video_1/005.mp4'
    from redpy.utils_redpy import write_video_file
    from moviepy.editor import ImageSequenceClip
    from glob import glob
    import os
    
    imgs = sorted(list(glob(f"{imgs_dir}/*.jpg")))
    video_clip = ImageSequenceClip(imgs, fps=1)
    os.makedirs(os.path.dirname(output_video_file), exist_ok=True)
    write_video_file(video_clip, output_video_file)



if __name__ == '__main__':
    # mix_pipeline_safetensors()
    # merge_imgs_to_video()
    pass




