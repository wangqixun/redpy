# 天空分割 grpc服务

输入| 输出可视化 
:-------------------------:|:-------------------------:
![img](test/data/tennis.png)  |  ![img](test/data/demo_output/sky_segmentation.jpg)

## server 地址
- **sit环境**
  - IP: 10.4.200.42
  - Port: 30125
- **prod环境**
  - 暂无

## 调用demo
python:
```python
from redpy.grpc import SkySegmentationClient
import cv2

if __name__ == '__main__':
    img = cv2.imread('/share/wangqixun/workspace/github_project/redpy/test/data/tennis.png')
    client = SkySegmentationClient(
        ip='10.4.200.42',
        port='30125',
    )
    res = client.run(img)
```

##### 负责人：guiwan











