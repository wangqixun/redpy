import grpc
import translate_pb2 as pb2
import translate_pb2_grpc as pb2_grpc
import pickle
import numpy as np
import pickle
import cv2

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
        self.stub = pb2_grpc.TranslateStub(channel)

    def run(self, inputs):
        # 调用 rpc 服务
        inputs_bytes = pickle.dumps(inputs)
        input_grpc = pb2.TranslateRequest(
            input_bytes=inputs_bytes,
        )
        response = self.stub.infer(input_grpc)
        result = pickle.loads(response.result_bytes)
        return result



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







