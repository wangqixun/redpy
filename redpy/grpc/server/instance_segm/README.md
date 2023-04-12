# 实例分割 grpc服务

输入| 输出可视化 
:-------------------------:|:-------------------------:
![img](test/data/skiing.png)  |  ![img](test/data/demo_output/instance_segm.jpg)


## server 地址
- **sit环境**
  - IP: 10.4.200.42
  - Port: 30123
- **prod环境**
  - 暂无

## 调用demo
python:
```python
from redpy.grpc import InstanceSegmClient
import cv2

if __name__ == '__main__':
    img = cv2.imread('/share/wangqixun/workspace/github_project/redpy/test/data/beauty.jpg')
    client = InstanceSegmClient(
        ip='10.4.200.42',
        port='30123',
    )
    res = client.run(
        img,
        sod=False,  # 主体检测
        sod_cfg=dict(  # 主体检测参数
            conf=0.4
        ),
    )
```

##### 负责人：guiwan










