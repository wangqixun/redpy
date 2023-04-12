# 翻译 grpc 服务

输入| 输出可视化 
:-------------------------:|:-------------------------:
  |  

## server 地址
- **sit环境**
  - IP: 10.4.200.42
  - Port: 30201
- **prod环境**
  - 暂无

## 调用demo
python:
```python
from redpy.grpc import TranslateClient
import cv2

if __name__ == '__main__':
    text = ['hello, what day is today?']
    client = Client(
        ip='10.4.200.42',
        port='30201',
        max_send_message_length=100, 
        max_receive_message_length=100,
    )

    res = client.run(
        text,
    )
    print(res)
```

##### 负责人：guiwan











