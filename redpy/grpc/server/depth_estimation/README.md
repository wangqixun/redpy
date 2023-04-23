# 深度估计 grpc服务

输入| 输出可视化 
:-------------------------:|:-------------------------:
![img](test/data/beauty.jpg)  |  ![img](test/data/demo_output/depth_estimation.jpg)

## server 地址
- **sit环境**
  - IP: 10.4.200.42
  - Port: 30301
- **prod环境**
  - 暂无

## 调用demo
python:
```python
import numpy as np
import cv2
import pickle
from redpy.grpc import CommonClient

if __name__ == '__main__':
    img = cv2.imread('/share/wangqixun/workspace/github_project/redpy/test/data/beauty.jpg')
    client = CommonClient(
        ip='10.4.200.42',
        port='30301',
        max_send_message_length=100, 
        max_receive_message_length=100,
    )
    res = client.run([img])
    print(res)
```

##### 负责人：guiwan











