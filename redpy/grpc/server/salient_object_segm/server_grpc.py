import grpc
import salient_object_segm_pb2 as pb2
import salient_object_segm_pb2_grpc as pb2_grpc
from concurrent import futures
import time
import pickle
from redpy.utils_redpy.logger_utils import setup_logger
logger = setup_logger(
    '/share/wangqixun/workspace/github_project/redpy/redpy/grpc/server/salient_object_segm/log.log',
    name='SalientObjectSegmServicer'
)


import sys
sys.path.append('/share/wangqixun/workspace/github_project/DIS/IS-Net')
import torch.nn.functional as F
from torchvision.transforms.functional import normalize
import torch
from models import *

import numpy as np
import cv2



# 实现 proto 文件中定义的 Servicer
class SalientObjectSegm(pb2_grpc.SalientObjectSegmServicer):
    def __init__(self):
        super(pb2_grpc.SalientObjectSegmServicer).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_size = [1024, 1024]

        model_path="/share/wangqixun/workspace/github_project/DIS/IS-Net/output/v5_fiou/checkpoint_epoch073_loss0.20714_fiou0.9069.pth.tar"  # the model path
        img_dir = '/share/wangqixun/workspace/github_project/DIS/demo_datasets/wqx/test_images'
        output_dir = '/share/wangqixun/workspace/github_project/DIS/demo_datasets/wqx/1024_v5'

        logger.info(f'loading model...')
        net = ISNetDIS()
        net.to(self.device)
        checkpoint = torch.load(model_path, map_location="cpu")
        net.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()})
        net.eval()

        self.net = net


    def infer(self, img):
        '''
            img: bgr channel
        '''
        im = img[..., ::-1] # RGB
        im_shp = im.shape[0:2]
        im_tensor = torch.tensor(np.copy(im), dtype=torch.float32).permute(2,0,1)
        im_tensor = F.upsample(torch.unsqueeze(im_tensor,0), self.input_size, mode="bilinear").type(torch.uint8)
        image = im_tensor / 255.
        image = normalize(image[0],[0.5,0.5,0.5],[1.0,1.0,1.0])[None]

        image = image.to(self.device)
        with torch.no_grad():
            result = self.net(image)
        result=torch.squeeze(F.upsample(result[0][0], im_shp, mode='bilinear'), 0)
        ma = torch.max(result)
        mi = torch.min(result)
        # result = (result-mi)/(ma-mi)
        if ma < 0.99:
            result *= 0
        result = result.permute(1,2,0).cpu().data.numpy()
        result = result * 255
        result = result.astype(np.uint8)

        return result

    # 实现 proto 文件中定义的 rpc 调用
    def infer_one_img(self, request, context):
        try:
            t1 = time.time()
            logger.info(f'=> bytes to array...')
            img = pickle.loads(request.img_bgr_bytes)
            t2 = time.time()
            logger.info(f'   img {img.shape} infering...')
            res_img = self.infer(img)
            t3 = time.time()
            logger.info(f'   [{t3-t2:.4f}] salient_object_segm infer finish.')
            res_bytes = pickle.dumps(res_img)
            output = pb2.SOSOneImgReply(
                result_bytes=res_bytes
            )
            return output
        except Exception as e:
            logger.error(f'=> infer: {e}')
            return None


def serve(port, max_send_message_length=256, max_receive_message_length=256):
    logger.info(f'rpc server: port={port}')
    # 启动 rpc 服务
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=2),
        options=[
            ('grpc.max_send_message_length', max_send_message_length * 1024 * 1024),
            ('grpc.max_receive_message_length', max_receive_message_length * 1024 * 1024)
        ],
    )
    pb2_grpc.add_SalientObjectSegmServicer_to_server(SalientObjectSegm(), server)
    server.add_insecure_port(f'[::]:{port}')
    server.start()
    logger.info('service start')
    try:
        while True:
            time.sleep(60*60*24) # one day in seconds
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == '__main__':
    serve(
        port=30126,
        max_send_message_length=256,  # 256m
        max_receive_message_length=256,  # 256m
    )

