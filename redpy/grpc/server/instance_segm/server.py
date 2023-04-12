from concurrent import futures
import time
import grpc
import pickle
from redpy.utils_redpy.logger_utils import setup_logger
logger = setup_logger(
    '/share/wangqixun/workspace/github_project/redpy/redpy/grpc/server/instance_segm/log.log',
    name='InstanceSegmServer'
)

import instance_segm_pb2 as pb2
import instance_segm_pb2_grpc as pb2_grpc
from mmdet.apis import inference_detector, init_detector
# import numpy as np
# import cv2

# 实现 proto 文件中定义的 GreeterServicer
class InstanceSegm(pb2_grpc.InstanceSegmServicer):
    def __init__(self):
        super(pb2_grpc.InstanceSegmServicer).__init__()
        cfg_path = '/share/wangqixun/workspace/github_project/CBNetV2_train/configs_slurm/refine_caslm/v20.py'
        weight_path = '/share/wangqixun/workspace/github_project/CBNetV2_train/new_data/v20/epoch_36.pth'
        pytorch_detector = init_detector(cfg_path, weight_path)

        self.model = pytorch_detector

    # 实现 proto 文件中定义的 rpc 调用
    def infer_one_img(self, request, context):
        try:
            t1 = time.time()
            t2 = time.time()
            img = pickle.loads(request.img_bgr_bytes)
            t3 = time.time()
            logger.info(f'=> [{t3-t2:.4f}] bytes to data.')
            logger.info(f'   img {img.shape} infering...')
            mmdet_res = inference_detector(self.model, img)
            t4 = time.time()
            logger.info(f'   [{t4-t3:.4f}] InstanceSegm infer finish.')
            mmdet_res_bytes = pickle.dumps(mmdet_res)
            output = pb2.ISOneImgReply(
                result_bytes=mmdet_res_bytes
            )

            # import cv2
            # img_vis = self.model.show_result(img, mmdet_res, score_thr=0.1)
            # cv2.imwrite('/share/wangqixun/workspace/github_project/redpy/test/data/demo_output/instance_segm.jpg', img_vis)

            return output
        except Exception as e:
            logger.error(f'=> infer: {e}')
            return None


def serve(port=30123, max_workers=10, max_send_message_length=256, max_receive_message_length=256):
    logger.info(f'rpc server: port={port}')
    # 启动 rpc 服务
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=max_workers),
        options=[
            ('grpc.max_send_message_length', max_send_message_length * 1024 * 1024),
            ('grpc.max_receive_message_length', max_receive_message_length * 1024 * 1024)
        ],
    )
    pb2_grpc.add_InstanceSegmServicer_to_server(InstanceSegm(), server)
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
        port=30123,
        max_workers=2,
        max_send_message_length=256,  # 256m
        max_receive_message_length=256,  # 256m
    )

