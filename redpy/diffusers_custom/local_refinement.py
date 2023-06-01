# import sys
# sys.path.append('/share/xingchuan/code/diffusion/improved_sd/sd_diffusers/workspace/controlnet_sam')
# from annotator.openpose import OpenposeDetector

import numpy as np
# import cv2
from PIL import Image
import PIL
import cv2

from .controlnet.pipeline_stable_diffusion_common import draw_kps

all = [
    'face_refinement'
]

def cv2_mix(img_pil, inpatinting_image, new_bbox):
    img = np.array(img_pil)
    face = np.array(inpatinting_image)
    mask = np.ones_like(face)*255
    x1, y1, x2, y2 = new_bbox
    center = (int((x1+x2)/2), int((y1+y2)/2))
    # import pdb;  pdb.set_trace()
    # cv2.imwrite('outputs/retro/img.png', img[:,:,::-1])
    # out = cv2.seamlessClone(face, img, mask, center, cv2.MIXED_CLONE)
    # cv2.imwrite('outputs/retro/mixed.png', out[:,:,::-1])
    # img[y1:y2, x1:x2] = face
    # out = cv2.seamlessClone(face, img, mask, center, cv2.MIXED_CLONE)
    # cv2.imwrite('outputs/retro/normal.png', out[:,:,::-1])

    out = cv2.seamlessClone(face, img, mask, center, cv2.MONOCHROME_TRANSFER)
    # cv2.imwrite('outputs/retro/transfer.png', out[:,:,::-1])
    # import pdb;  pdb.set_trace()
    out = Image.fromarray(out)
    return out


def face_refinement(img_pil, pipe, bbox, kps, embs, control_index=None, gender=None, age=None, prompt=None, bbox_img_expend_ratio=0.65, bbox_inpainting_expend_ratio=0.2, **kwargs):
    if control_index is None:
        for idx, p in enumerate(pipe.controlnet_list):
            if p.__class__.__name__ == 'VisualControlNetModel':
                control_index = idx

    assert control_index is not None, "There is no Face-ControlNet in the pipe. If you need, please contacting guiwan"

    face_emb = embs
    face_kps = kps
    face_bbox = bbox
    gender = gender
    age = age

    width_img, height_img = img_pil.size

    # bbox 外扩
    bbox = np.array(face_bbox)
    x1, y1, x2, y2 = bbox
    bbox = (int(x1), int(np.ceil(y1)), int(x2), int(np.ceil(y2)))
    w = x2 - x1
    h = y2 - y1
    x1 = int(np.clip(x1 - w * bbox_img_expend_ratio, 0, width_img-1))
    y1 = int(np.clip(y1 - h * bbox_img_expend_ratio, 0, height_img-1))
    x2 = int(np.ceil(np.clip(x2 + w * bbox_img_expend_ratio, 0, width_img-1)))
    y2 = int(np.ceil(np.clip(y2 + h * bbox_img_expend_ratio, 0, height_img-1)))
    new_bbox = (x1, y1, x2, y2)

    # crop 外扩的并 resize 到长边 512
    new_bbox_image = img_pil.crop(new_bbox)
    resize_new_bbox_image = new_bbox_image.copy()
    w, h = resize_new_bbox_image.size
    ratio = 512 / max(h, w)
    w_resize_new = (round(ratio * w) // 8) * 8
    h_resize_new = (round(ratio * h) // 8) * 8
    resize_new_bbox_image = resize_new_bbox_image.resize([w_resize_new, h_resize_new])
    # resize_new_bbox_image.save('/share/wangqixun/workspace/tmp/0002.jpg')

    # mask image
    mask_image = np.zeros_like(img_pil)
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    inpatinting_x1 = int(np.clip(x1 - w * bbox_inpainting_expend_ratio, 0, width_img-1))
    inpatinting_y1 = int(np.clip(y1 - h * bbox_inpainting_expend_ratio, 0, height_img-1))
    inpatinting_x2 = int(np.ceil(np.clip(x2 + w * bbox_inpainting_expend_ratio, 0, width_img-1)))
    inpatinting_y2 = int(np.ceil(np.clip(y2 + h * bbox_inpainting_expend_ratio, 0, height_img-1)))
    mask_image[inpatinting_y1:inpatinting_y2,inpatinting_x1:inpatinting_x2] = 255
    mask_image = mask_image[new_bbox[1]:new_bbox[3], new_bbox[0]:new_bbox[2]]
    mask_image = Image.fromarray(mask_image)
    mask_image = mask_image.resize(resize_new_bbox_image.size, resample=PIL.Image.NEAREST)
    # mask_image.save('/share/wangqixun/workspace/tmp/0003.jpg')

    # face control image
    crop_face_kps = np.array(face_kps).reshape([-1, 2])
    crop_face_kps[:, ::2] = crop_face_kps[:, ::2] - new_bbox[0]
    crop_face_kps[:, 1::2] = crop_face_kps[:, 1::2] - new_bbox[1]
    # crop_face_kps = crop_face_kps * ratio
    crop_face_kps[:, ::2] = crop_face_kps[:, ::2] * (w_resize_new / new_bbox_image.size[0])
    crop_face_kps[:, 1::2] = crop_face_kps[:, 1::2] * (h_resize_new / new_bbox_image.size[1])
    control_image_face = draw_kps(resize_new_bbox_image, crop_face_kps)
    # control_image_face.save('/share/wangqixun/workspace/tmp/0004.jpg')

    # controlnet_conditioning
    controlnet_conditioning = [
        dict(
            control_image=control_image_face,
            control_index=control_index,
            control_weight=0.8,
            control_visual_emb=face_emb,
        ),
    ]

    # prompt
    if gender == 1:
        gender_prompt = 'man'
    elif gender == 0:
        gender_prompt = 'girl'
    else:
        gender_prompt = 'person'
    if age is not None:
        person_prompt = f"a {age} year old {gender_prompt}"
    else:
        person_prompt = f"a {gender_prompt}"
    face_prompt = f"{prompt if prompt is not None else ''} {person_prompt}"

    # gogogo
    inpatinting_image = pipe.inpainting(
        image=resize_new_bbox_image,
        mask_image=mask_image,
        controlnet_conditioning=controlnet_conditioning,
        prompt=face_prompt,
        **kwargs
    ).images[0]
    # inpatinting_image.save('/share/wangqixun/workspace/tmp/0005.jpg')

    # img_pil.paste(inpatinting_image.resize((new_bbox[2]-new_bbox[0], new_bbox[3]-new_bbox[1])), new_bbox)
    # inpatinting_image = inpatinting_image.resize((new_bbox[2]-new_bbox[0], new_bbox[3]-new_bbox[1]))
    # img_pil = cv2_mix(img_pil, inpatinting_image, new_bbox)

    mask_image = np.zeros_like(img_pil)
    mask_image[new_bbox[1]:new_bbox[3]+1, new_bbox[0]:new_bbox[2]+1] = 255
    mask_image = cv2.GaussianBlur(mask_image, (91, 91), 0) 
    # cv2.imwrite('/share/wangqixun/workspace/tmp/0006.jpg', mask_image)
    mask_image = mask_image / 255.
    new_image = img_pil.copy()
    new_image.paste(inpatinting_image.resize((new_bbox[2]-new_bbox[0], new_bbox[3]-new_bbox[1])), new_bbox)
    new_image = np.array(new_image) * (mask_image) + np.array(img_pil) * (1 - mask_image)
    new_image = Image.fromarray(new_image.astype(np.uint8))
    img_pil = new_image

    

    # img_pil.save('/share/wangqixun/workspace/tmp/0007.jpg')
    # xxxx

    return img_pil

    






