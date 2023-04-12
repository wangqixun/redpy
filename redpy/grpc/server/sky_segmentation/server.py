import grpc
import sky_segmentation_pb2 as pb2
import sky_segmentation_pb2_grpc as pb2_grpc
from concurrent import futures
import time
import pickle
from redpy.utils_redpy.logger_utils import setup_logger
logger = setup_logger(
    '/share/wangqixun/workspace/github_project/redpy/redpy/grpc/server/sky_segmentation/log.log',
    name='SkySegmentationServer'
)

import onnxruntime
import cv2
import copy
import numpy as np

# 实现 proto 文件中定义的 Servicer
class SkySegmentation(pb2_grpc.SkySegmentationServicer):
    def __init__(self):
        super(pb2_grpc.SkySegmentationServicer).__init__()
        ckp_file = '/share/yaoyifan/workspace/Sky-Segmentation-and-Post-processing/v4_ft.onnx'
        logger.info('loading model ...')
        onnx_session = onnxruntime.InferenceSession(ckp_file, providers=onnxruntime.get_available_providers())
        self.onnx_session = onnx_session

    def run_inference(self, img, input_size=[320, 320]):
        '''
            img: bgr
        '''
        # Pre process:Resize, BGR->RGB, Transpose, PyTorch standardization, float32 cast
        temp_image = copy.deepcopy(img)
        resize_image = cv2.resize(temp_image, dsize=(input_size[0], input_size[1]))
        x = cv2.cvtColor(resize_image, cv2.COLOR_BGR2RGB)
        x = np.array(x, dtype=np.float32)
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        x = (x / 255 - mean) / std
        x = x.transpose(2, 0, 1)
        x = x.reshape(-1, 3, input_size[0], input_size[1]).astype('float32')

        # Inference
        input_name = self.onnx_session.get_inputs()[0].name
        output_name = self.onnx_session.get_outputs()[0].name
        onnx_result = self.onnx_session.run([output_name], {input_name: x})

        # Post process
        onnx_result = np.array(onnx_result).squeeze()
        min_value = np.min(onnx_result)
        max_value = np.max(onnx_result)
        # onnx_result = (onnx_result - min_value) / (max_value - min_value)
        onnx_result *= 255
        onnx_result = onnx_result.astype('uint8')

        onnx_result = cv2.resize(onnx_result, (img.shape[1], img.shape[0]))

        return onnx_result

    # 实现 proto 文件中定义的 rpc 调用
    def infer_one_img(self, request, context):
        try:
            t1 = time.time()
            logger.info(f'=> bytes to array...')
            img = pickle.loads(request.img_bgr_bytes)
            t2 = time.time()
            logger.info(f'   img {img.shape} infering...')
            res_img = self.run_inference(img, input_size=[320, 320])
            t3 = time.time()
            logger.info(f'   [{t3-t2:.4f}] sky segmentation infer finish.')
            res_bytes = pickle.dumps(res_img)
            output = pb2.SSOneImgReply(
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
    pb2_grpc.add_SkySegmentationServicer_to_server(SkySegmentation(), server)
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
        port=30125,
        max_send_message_length=256,  # 256m
        max_receive_message_length=256,  # 256m
    )

