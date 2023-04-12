import grpc
import salient_object_segm_pb2 as pb2
import salient_object_segm_pb2_grpc as pb2_grpc
import pickle
import numpy as np
import cv2
import pickle

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
        stub = pb2_grpc.SalientObjectSegmStub(channel)

        self.stub = stub


    def run(self, img):
        # 调用 rpc 服务
        img_bgr_bytes = pickle.dumps(img)
        input_grpc = pb2.SOSOneImgRequest(
            img_bgr_bytes=img_bgr_bytes,
        )
        response = self.stub.infer_one_img(input_grpc)
        result = pickle.loads(response.result_bytes)
        return result

        

if __name__ == '__main__':
    img = cv2.imread('/share/wangqixun/workspace/github_project/redpy/test/data/beauty.jpg')
    client = Client(
        ip='10.4.200.42',
        port='30126',
        max_send_message_length=100, 
        max_receive_message_length=100,
    )

    res = client.run(img)







