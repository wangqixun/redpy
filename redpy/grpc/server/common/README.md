# 通用转换 gRPC 服务

把任意函数转换成对应的 gRPC 服务


## 使用方法
#### 1. 新建一个 xxxx.py，写一个自定义class，其中包含一个功能函数，比如：
```python
import time
import numpy as np
import queue


class Demo():
    def __init__(self, maxsize=4):
        self.queue = queue.Queue(maxsize)
        for i in range(maxsize):
            model = f"model_{i}"
            self.queue.put(model)

    def infer(self, a, b=1, c=2):
        model = self.queue.get(True, timeout=10)
        time.sleep(np.random.randint(3))
        res = f"{model},{a},{b},{c}"
        print(res)
        self.queue.put(model)
        return res


if __name__ == "__main__":
    service = Demo(maxsize=4)
    res = service.infer('test a=a')
```

保证其可顺利运行出结果

``` shell
python xxxx.py
```

#### 2. 对功能函数进行修饰，其它不变。比如：
```python
from redpy.grpc.server.common.server import convert_to_server

...
...

    @convert_to_server(server_name='DemoServer', port=12345, max_workers=10)
    def infer(self, a, b=1, c=2):
        model = self.queue.get(True, timeout=10)
        time.sleep(np.random.randint(3))
        res = f"{model},{a},{b},{c}"
        print(res)
        self.queue.put(model)
        return res

...
...
```

在终端重新启动```python xxxx.py```即可

#### 3. 客户端 client 调用如下，其中 ip 地址通过 ```ifconfig``` 查看：
```python
from redpy.grpc import CommonClient

a = 'aaa'
b = 123
c = '!@#$%^&*()_+<>?}{":'
client = Client(
    ip='xx.x.xxx.xx',
    port='12345',
    max_send_message_length=100, 
    max_receive_message_length=100,
)
res = client.run([a, b, c])
```


##### 负责人：guiwan











