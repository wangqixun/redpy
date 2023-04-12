import queue
from redpy.grpc.server.common.server import convert_to_server
import time
import numpy as np

from IPython import embed


class Demo():
    def __init__(self, maxsize=4):
        self.queue = queue.Queue(maxsize)
        for i in range(maxsize):
            model = f"model_{i}"
            self.queue.put(model)


    @convert_to_server(server_name='DemoServer', port=30129, max_workers=10)
    def infer(self, a, b=1, c=2):
        model = self.queue.get(True, timeout=10)
        time.sleep(np.random.randint(3))
        res = f"{model},{a},{b},{c}"
        print(res)
        self.queue.put(model)
        return res


if __name__ == "__main__":
    service = Demo(maxsize=4)
    res = service.infer('a')







