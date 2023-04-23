# 姿态估计 grpc 服务


## server 地址
- **sit环境**
  - IP: 10.4.200.42
  - Port: 12358
- **prod环境**
  - 暂无

## 调用demo
python:
```python
from redpy.grpc import CommonClient
import cv2

client = CommonClient(
    ip='10.4.200.42',
    port='12358',
    max_send_message_length=100,
    max_receive_message_length=100
    )

img = cv2.imread("/share/xingchuan/code/onlinedemos/open-pose/test.jpg")
input_format = "BGR"
output_format = "BGR"

# input_format/output_format 选择RGB/BGR，默认BGR
res = client.run([img, input_format, output_format])
cv2.imwrite("1.jpg", res)
```

##### 负责人：xingchuan
