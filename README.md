# RedPy
（听说还挺好用的

## Install
```
pip install git+https://code.devops.xiaohongshu.com/wangqixun/redpy.git@stable
```

或者 clone 下来手动安装

```
git clone -b stable https://code.devops.xiaohongshu.com/wangqixun/redpy.git
cd redpy
pip install -e .
```



## gRPC服务

若无特殊说明，涉及到图像的输入默认使用BGR通道

+ [**grpc服务快速部署**](redpy/grpc/server/common)

+ [**实例分割**](redpy/grpc/server/instance_segm)

+ [**天空分割**](redpy/grpc/server/sky_segmentation)

+ [**深度估计**](redpy/grpc/server/depth_estimation)

+ [**主体分割**](redpy/grpc/server/salient_object_segm)

+ [**人脸信息**](redpy/grpc/server/arcface)

+ [**BLIP2图像打标**](redpy/grpc/server/blip2)

+ [**英译汉**](redpy/grpc/server/translate)


## diffusers 定制化

+ [**ControlNet**](redpy/diffusers_custom/controlnet)

+ [**Lora**](redpy/diffusers_custom/lora)

+ [**Textual Inversion**](redpy/diffusers_custom/textual_inversion)



## 工具
+ [**常用工具箱**](redpy/utils_redpy)

