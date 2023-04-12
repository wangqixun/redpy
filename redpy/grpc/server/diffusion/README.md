# 天空分割 grpc服务

输入| 输出可视化 
:-------------------------:|:-------------------------:
![img](test/data/tennis.pngxxx)  |  ![img](test/data/demo_output/sky_segmentation.jpgxxxx)

## server 地址
- **sit环境**
  - IP: 10.4.200.42
  - Port: 30128-30132
- **prod环境**
  - 暂无

## 调用demo
python:
```python
from redpy.grpc import DiffusionClient
import cv2


if __name__ == '__main__':
  text = '美丽的动漫女性肖像，亚洲人，娃娃脸，穿着黑色连帽衫，干净的脸，枯死的眼睛，棕色的头发，对称的面部，艺术站，人物概念艺术，迪斯尼海报，2d插图，逼真的脸，高细节，电影，8k，8k高清，平面设计，--ar 3:4-uplight'

  text = 'portrait of beautiful anime female, asian, baby face, wearing a black hoodie, clean face, dead eyes, brown hair, symmetrical facial, artstation, character concept art, Disney poster, 2d illustration, realistic face, high detail, Cinematic, 8k, 8k hd, graphic design, --ar 3:4 --uplight'

  client = DiffusionClient(
      ip='10.4.200.42',
      port='30128', # 30128-30132 均可
      max_send_message_length=100, 
      max_receive_message_length=100,
  )

  res = client.run(
      text,
  )
```

##### 负责人：guiwan











