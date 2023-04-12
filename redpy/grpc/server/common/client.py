import grpc
import common_pb2 as pb2
import common_pb2_grpc as pb2_grpc
import pickle
import numpy as np
import pickle
import cv2
import time
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED, ALL_COMPLETED


__all__ = [
    'Client'
]

class Client():
    def __init__(self, ip, port, max_send_message_length=100, max_receive_message_length=100):
        # 连接 rpc 服务器
        channel = grpc.insecure_channel(
            f'{ip}:{port}',
            options=[
                ('grpc.max_send_message_length', max_send_message_length * 1024 * 1024),
                ('grpc.max_receive_message_length', max_receive_message_length * 1024 * 1024)
            ]
        )
        self.stub = pb2_grpc.CommonStub(channel)

    def run(self, input):
        # 调用 rpc 服务
        input_bytes = pickle.dumps(input)
        input_grpc = pb2.CommonRequest(input_bytes=input_bytes)
        response = self.stub.common_infer(input_grpc)
        result = pickle.loads(response.result_bytes)
        return result


def infer(i):
    a = str(i)
    b = str(i)
    c = str(i)
    client = Client(
        ip='10.4.200.42',
        port='30129',
        max_send_message_length=100, 
        max_receive_message_length=100,
    )
    res = client.run([a, b, c])
    # cv2.imwrite('./test_img.jpg', res)
    print(res)


if __name__ == '__main__':
    # img = cv2.imread('/share/wangqixun/workspace/github_project/redpy/test/data/beauty.jpg')

    t1 = time.time()
    with ThreadPoolExecutor(max_workers=10) as t:
        all_task = [t.submit(infer, i) for i in range(50)]
        wait(all_task, return_when=ALL_COMPLETED)
    t2 = time.time()

    total = 50
    cost = t2 - t1
    print("cost: {}, total: {}, qps: {}".format(cost, total, (total/cost)))





