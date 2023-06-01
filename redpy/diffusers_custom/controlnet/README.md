# Stable Diffusion Common Pipeline

基于 Diffusers 开发，集成多control、text2img、img2img、inpainting，兼容原版 Diffusers pipeline 参数。Diffusers 已有功能不进行集成。

## Load Pipeline
```python
from redpy.diffusers_custom import StableDiffusionCommonPipeline

# load pipeline
base_model = 'base_model path'
pipe = StableDiffusionCommonPipeline.from_pretrained(base_model, safety_checker=None, feature_extractor=None)
```

## Use VAE-Fp32-Other-Fp16
```python
pipe = pipe.cuda_half()
```

## Use Text2img
```python
from PIL import Image

# infer, 兼容原版 diffuser pipeline 参数，比如 num_inference_steps、guidance_scale 等
image = pipe.text2img(
    prompt="your prompt",
    negative_prompt="your negative_prompt",
    height=512,
    width=512,
).images[0]
```


## Use Img2img
```python
from PIL import Image

# image
input_image = Image.open('input_image path').convert('RGB')

# infer, 兼容原版 diffuser pipeline 参数，比如 num_inference_steps、guidance_scale 等
image = pipe.img2img(
    image=input_image,
    prompt="your prompt",
    negative_prompt="your negative_prompt",
).images[0]
```


## Use Inpainting
```python
from PIL import Image

# image and mask, mask 白色(255) 部分代表 inpainting 补充部分
input_image = Image.open('input_image path').convert('RGB')
mask_image = Image.open('mask_image path').convert('RGB')

# infer, 兼容原版 diffuser pipeline 参数，比如 num_inference_steps、guidance_scale 等
image = pipe.inpainting(
    image=input_image,
    mask_image=mask_image,
    prompt="your prompt",
    negative_prompt="your negative_prompt",
).images[0]
```


## Use Multi-Controlnet
```python
from PIL import Image
from redpy.diffusers_custom import StableDiffusionCommonPipeline

# load pipeline
base_model = 'base_model path'
controlnet_path_list = ['controlnet-canny path', "controlnet-depth path"]
pipe = StableDiffusionCommonPipeline.from_pretrained(base_model, controlnet_list=controlnet_path_list, safety_checker=None, feature_extractor=None)

# image 
input_image = Image.open("input_image path").convert('RGB')
control_image_canny = Image.open("control_image_canny path").convert('RGB')
control_image_depth = Image.open("control_image_depth path").convert('RGB')

# controlnet infer 配置，controlnet_conditioning 中成员顺序可打乱
controlnet_conditioning = [
    dict(
        control_index=0,  # controlnet 标识，对应 controlnet_path_list 中脚标
        control_image=control_image_canny,  # 对应的 controlnet 输入图像
        control_weight=1.1,  # 对应的 controlnet 权重
    ),
    dict(
        control_index=1,  # controlnet 标识，对应 controlnet_path_list 中脚标
        control_image=control_image_depth,  # 对应的 controlnet 输入图像
        control_weight=0.9,  # 对应的 controlnet 权重
    ),
]

# infer, text2img、img2img、inpainting 均可
image = pipe.img2img(
    image=input_image,
    controlnet_conditioning=controlnet_conditioning,
    prompt="your prompt",
    negative_prompt="your negative_prompt",
).images[0]

```



