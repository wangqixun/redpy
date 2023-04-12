# RedPy
（听说还挺好用的

## Install
```
pip install git+https://github.com/wangqixun/redpy.git@stable
```

或者 clone 下来手动安装

```
git clone -b stable https://github.com/wangqixun/redpy.git
cd redpy
pip install -e .
```



## gRPC服务

若无特殊说明，涉及到图像的输入默认使用BGR通道

+ [**grpc服务快速部署**](redpy/grpc/server/common)

+ [**人脸信息**](redpy/grpc/server/arcface)



## diffusers 定制化

+ [**ControlNet**](redpy/diffusers_custom/controlnet)

+ [**Lora**](redpy/diffusers_custom/lora)

+ [**Textual Inversion**](redpy/diffusers_custom/textual_inversion)



## 工具
+ [**常用工具箱**](redpy/utils_redpy)

