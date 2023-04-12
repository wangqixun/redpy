import grpc
import diffusion_pb2 as pb2
import diffusion_pb2_grpc as pb2_grpc
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
        self.stub = pb2_grpc.DiffusionStub(channel)

    def run(self, text):
        # 调用 rpc 服务
        text_bytes = pickle.dumps(text)
        input_grpc = pb2.DOneTextRequest(
            input_bytes=text_bytes,
        )
        response = self.stub.infer_one_text(input_grpc)
        result = pickle.loads(response.result_bytes)
        return result

        

if __name__ == '__main__':
    # img = cv2.imread('/share/wangqixun/workspace/github_project/redpy/test/data/beauty.jpg')
    text = '美丽的女性肖像，亚洲人，娃娃脸，穿着黑色连帽衫，干净的脸，枯死的眼睛，棕色的头发，对称的面部，艺术，人物概念艺术，迪斯尼海报，逼真的脸，高细节，电影，8k，8k高清，平面设计，--ar 3:4-uplight'
    text = 'portrait of beautiful female, asian, baby face, wearing a black hoodie, clean face, dead eyes, brown hair, symmetrical facial, artstation, character concept art, Disney poster, realistic face, high detail, Cinematic, 8k, 8k hd, graphic design, --ar 3:4 --uplight'
    client = Client(
        ip='10.4.200.42',
        port='30133',
        max_send_message_length=100, 
        max_receive_message_length=100,
    )

    res = client.run(
        text,
    )
    cv2.imwrite('./test_img.jpg', res)







