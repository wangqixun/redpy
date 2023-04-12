import grpc
import translate_pb2 as pb2
import translate_pb2_grpc as pb2_grpc
from concurrent import futures
import time
import pickle
from redpy.utils_redpy.logger_utils import setup_logger
logger = setup_logger(
    '/share/wangqixun/workspace/github_project/redpy/redpy/grpc/server/translate/log.log',
    name='TranslateEnToZhServer'
)

import cv2
import copy
import numpy as np
from transformers import AutoModelWithLMHead, AutoTokenizer, pipeline

# 实现 proto 文件中定义的 Servicer
class Translate(pb2_grpc.TranslateServicer):
    def __init__(self):
        super(pb2_grpc.TranslateServicer).__init__()
        self.init()


    # TODO
    def init(self, ):
        self.model_info = dict(
            model_name='translate_en-to-zh',
        )
        model_path = '/share/wangqixun/workspace/github_project/transformers_checkpoint/trans-opus-mt-en-zh'
        model = AutoModelWithLMHead.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.translation = pipeline("translation_en_to_zh", model=model, tokenizer=tokenizer, device=0)


    # TODO
    def _infer(self, inputs):
        prompts_en = inputs
        prompts_zh = self.translation(prompts_en, max_length=500)
        return prompts_zh


    # 实现 proto 文件中定义的 rpc 调用
    def infer(self, request, context):
        try:
            t1 = time.time()
            logger.info(f'=> bytes to array ...')
            inputs = pickle.loads(request.input_bytes)
            t2 = time.time()
            logger.info(f'   model infering ...')
            res = self._infer(inputs)
            t3 = time.time()
            logger.info(f'   [{t3-t2:.4f}] {self.model_info["model_name"]} infer finish.')
            result_bytes = pickle.dumps(res)
            output = pb2.TranslateReply(
                result_bytes=result_bytes
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
    pb2_grpc.add_TranslateServicer_to_server(Translate(), server)
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
        port=30201,
        max_send_message_length=256,  # 256m
        max_receive_message_length=256,  # 256m
    )

