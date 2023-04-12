# 主体分割 grpc服务

输入| 输出可视化 
:-------------------------:|:-------------------------:
![img](test/data/beauty.jpg)  |  ![img](test/data/demo_output/salient_object_segm.jpg)

## server 地址
- **sit环境**
  - IP: 10.4.200.43
  - Port: 30126
- **prod环境**
  - 暂无

## 调用demo
python:
```python
from redpy.grpc import CommonClient
import cv2

if __name__ == '__main__':
    img = cv2.imread('/share/wangqixun/workspace/github_project/redpy/test/data/beauty.jpg')
    client = CommonClient(
        ip='10.4.200.43',
        port='30126',
    )
    res = client.run([img])
```

##### 负责人：guiwan










