import grpc
import diffusion_pb2 as pb2
import diffusion_pb2_grpc as pb2_grpc
from concurrent import futures
import time
import pickle
from redpy.utils_redpy.logger_utils import setup_logger
logger = setup_logger(
    '/share/wangqixun/workspace/github_project/redpy/redpy/grpc/server/diffusion/log/mj_log.log',
    name='midjourneyDiffusionServer'
)

import cv2
import numpy as np
from diffusers import StableDiffusionPipeline
import torch



# 实现 proto 文件中定义的 Servicer
class Diffusion(pb2_grpc.DiffusionServicer):
    def __init__(self):
        super(pb2_grpc.DiffusionServicer).__init__()
        self.pipe = StableDiffusionPipeline.from_pretrained(
            # "/share/wangqixun/workspace/github_project/diffusers/checkpoint/midjourney_v0_50000",
            "/share/wangqixun/workspace/github_project/diffusers/checkpoint/midjourney_v1_20000",
            torch_dtype=torch.float16,
        ).to("cuda")


    def run_inference(self, text):
        '''
            text

        '''
        prompt = text
        image = self.pipe(prompt, guidance_scale=7.5).images[0]  

        res_img = np.array(image)[..., ::-1]
        return res_img


    # 实现 proto 文件中定义的 rpc 调用
    def infer_one_text(self, request, context):
        try:
            t1 = time.time()
            logger.info(f'=> bytes to text...')
            text = pickle.loads(request.input_bytes)
            t2 = time.time()
            logger.info(f'   {text} infering...')
            res_img = self.run_inference(text)
            t3 = time.time()
            logger.info(f'   [{t3-t2:.4f}] midjourney model infer finish.')
            res_bytes = pickle.dumps(res_img)
            output = pb2.DOneTextReply(
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
    pb2_grpc.add_DiffusionServicer_to_server(Diffusion(), server)
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
        port=30133,
        max_send_message_length=256,  # 256m
        max_receive_message_length=256,  # 256m
    )

