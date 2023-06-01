







## Face Refinement

人脸修复，多用于小人脸丑陋、模糊的修复。大人脸也可以用，但是感觉没太大必要

<div align=center>
<img src="../../test/data/readme/face_refinement/face_refinement_pipeline2.png" width = "750" />
</div>

使用方法

```python
from redpy.diffusers_custom import StableDiffusionCommonPipeline
from redpy.diffusers_custom import face_refinement
from redpy.grpc import CommonClient
from PIL import Image

# load pipeline
# Face Refinement 需要 Face Controlnet 支持, 确保controlnet中存在
base_model = 'base_model path'
controlnet_path_list = ['controlnet-face path', "other controlnet-xxx"]
pipe = StableDiffusionCommonPipeline.from_pretrained(base_model, controlnet_list=controlnet_path_list, safety_checker=None, feature_extractor=None)

# input_image
input_image = Image.open('image file path').convert('RGB')

# step 1. 输出一个 output image
image = pipe.img2img(
    image=input_image,
    strength=0.6,
    prompt='your prompt',
    negative_prompt='your negative_prompt',
).images[0]

# step 2. 准备人脸信息，人脸服务已经部署完毕
client_face = CommonClient(ip='x.x.x.x', port='xxxx')
control_image_bgr = np.array(input_image)[..., ::-1]
face_info = client_face.run([control_image_bgr])

# step 3. 人脸修复
for idx in range(len(face_info)):
    face_emb = face_info[idx]['embedding']
    face_kps = face_info[idx]['kps']
    face_bbox = face_info[idx]['bbox']
    gender = face_info[idx]['gender']
    age = face_info[idx]['age']

    image = face_refinement(
        img_pil=image, 
        pipe=pipe, 
        bbox=face_bbox, 
        kps=face_kps, 
        embs=face_emb, 
        gender=gender, 
        age=age, 
        prompt=prompt, 
        negative_prompt=negative_prompt,
        strength=0.6,
        guidance_scale=5.,
        num_inference_steps=30,
    )
```

一些例子

input| img2img | face crop | face kps | face refinement | output 
:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:
![img](../../test/data/readme/face_refinement/11.jpeg)|![img](../../test/data/readme/face_refinement/12.jpeg)|![img](../../test/data/readme/face_refinement/13.jpeg)|![img](../../test/data/readme/face_refinement/14.jpeg)|![img](../../test/data/readme/face_refinement/15.jpeg)|![img](../../test/data/readme/face_refinement/16.jpeg)
![img](../../test/data/readme/face_refinement/21.jpeg)|![img](../../test/data/readme/face_refinement/22.jpeg)|![img](../../test/data/readme/face_refinement/23.jpeg)|![img](../../test/data/readme/face_refinement/24.jpeg)|![img](../../test/data/readme/face_refinement/25.jpeg)|![img](../../test/data/readme/face_refinement/26.jpeg)
![img](../../test/data/readme/face_refinement/31.jpeg)|![img](../../test/data/readme/face_refinement/32.jpeg)|![img](../../test/data/readme/face_refinement/33.jpeg)|![img](../../test/data/readme/face_refinement/34.jpeg)|![img](../../test/data/readme/face_refinement/35.jpeg)|![img](../../test/data/readme/face_refinement/36.jpeg)
![img](../../test/data/readme/face_refinement/41.jpeg)|![img](../../test/data/readme/face_refinement/42.jpeg)|![img](../../test/data/readme/face_refinement/43.jpeg)|![img](../../test/data/readme/face_refinement/44.jpeg)|![img](../../test/data/readme/face_refinement/45.jpeg)|![img](../../test/data/readme/face_refinement/46.jpeg)
![img](../../test/data/readme/face_refinement/51.jpeg)|![img](../../test/data/readme/face_refinement/52.jpeg)|![img](../../test/data/readme/face_refinement/53.jpeg)|![img](../../test/data/readme/face_refinement/54.jpeg)|![img](../../test/data/readme/face_refinement/55.jpeg)|![img](../../test/data/readme/face_refinement/56.jpeg)
![img](../../test/data/readme/face_refinement/61.jpeg)|![img](../../test/data/readme/face_refinement/62.jpeg)|![img](../../test/data/readme/face_refinement/63.jpeg)|![img](../../test/data/readme/face_refinement/64.jpeg)|![img](../../test/data/readme/face_refinement/65.jpeg)|![img](../../test/data/readme/face_refinement/66.jpeg)
![img](../../test/data/readme/face_refinement/71.jpeg)|![img](../../test/data/readme/face_refinement/72.jpeg)|![img](../../test/data/readme/face_refinement/73.jpeg)|![img](../../test/data/readme/face_refinement/74.jpeg)|![img](../../test/data/readme/face_refinement/75.jpeg)|![img](../../test/data/readme/face_refinement/76.jpeg)
![img](../../test/data/readme/face_refinement/81.jpeg)|![img](../../test/data/readme/face_refinement/82.jpeg)|![img](../../test/data/readme/face_refinement/83.jpeg)|![img](../../test/data/readme/face_refinement/84.jpeg)|![img](../../test/data/readme/face_refinement/85.jpeg)|![img](../../test/data/readme/face_refinement/86.jpeg)



