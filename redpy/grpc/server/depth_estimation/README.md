# 深度估计 grpc服务

输入| 输出可视化 
:-------------------------:|:-------------------------:
![img](test/data/beauty.jpg)  |  ![img](test/data/demo_output/depth_estimation.jpg)

## server 地址
- **sit环境**
  - IP: 10.4.200.42
  - Port: 30124
- **prod环境**
  - 暂无

## 调用demo
python:
```python
from redpy.grpc import DepthEstimationClient
import cv2

if __name__ == '__main__':
    img = cv2.imread('/share/wangqixun/workspace/github_project/redpy/test/data/beauty.jpg')
    client = DepthEstimationClient(
        ip='10.4.200.42',
        port='30124',
    )
    res = client.run(img)
```

##### 负责人：guiwan











