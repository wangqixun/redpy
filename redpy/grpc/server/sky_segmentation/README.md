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
import cv2
from redpy.grpc import CommonClient

image = cv2.imread('/share/wangqixun/workspace/github_project/redpy/test/data/WX20230420-175738@2x.png')
client = CommonClient(
    ip='10.4.200.42',
    port='30125',
    max_send_message_length=100, 
    max_receive_message_length=100,
)
res = client.run([image])
```

##### 负责人：guiwan











