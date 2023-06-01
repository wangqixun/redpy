import grpc
import common_pb2 as pb2
import common_pb2_grpc as pb2_grpc

from functools import wraps
from concurrent import futures
import time
import pickle
import os
import traceback
# from IPython import embed

from redpy.utils_redpy.logger_utils import setup_logger



def convert_to_server(server_name, port, max_workers=1):
    tmp_dir = '/tmp/redpy_log/'
    os.makedirs(tmp_dir, exist_ok=True)
    logger = setup_logger(
        os.path.join(tmp_dir, f"{server_name}.log"),
        name=server_name,
    )


    def serve(servicer, max_send_message_length=256, max_receive_message_length=256):
        logger.info(f'rpc server: port={port}')
        # 启动 rpc 服务
        server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=max_workers),
            options=[
                ('grpc.max_send_message_length', max_send_message_length * 1024 * 1024),
                ('grpc.max_receive_message_length', max_receive_message_length * 1024 * 1024)
            ],
        )
        pb2_grpc.add_CommonServicer_to_server(servicer, server)
        server.add_insecure_port(f'[::]:{port}')
        server.start()
        logger.info(f'{server_name} service start')
        try:
            while True:
                time.sleep(60*60*24) # one day in seconds
        except KeyboardInterrupt:
            server.stop(0)


    def _convert_to_server(func):
        raw_func = func


        class Servicer():
            def __init__(self, servicer) -> None:
                # for k, v in servicer.__dict__.items():
                #     self.__setattr__(k, v)
                logger.info('继承成员：')
                for k in dir(servicer):
                    if k.startswith('__'):
                        continue
                    v = getattr(servicer, k)
                    # if not callable(v):
                    #     continue
                    logger.info(k)
                    self.__setattr__(k, v)

            def common_infer(self, request, context):
                try:
                    t1 = time.time()
                    logger.info(f'=> {server_name} bytes to data ...')
                    input = pickle.loads(request.input_bytes)
                    t2 = time.time()
                    logger.info(f'   {server_name} infering ...')
                    result = raw_func(self, *input)
                    t3 = time.time()
                    logger.info(f'   {server_name} [{t3-t2:.4f}] infer finish.')
                except Exception as e:
                    logger.error(f'   {traceback.format_exc()}')
                    result = None
                result_bytes = pickle.dumps(result)
                output = pb2.CommonReply(result_bytes=result_bytes)
                return output


        @wraps(func)
        def wrapper(*args, **kwargs):
            # print('convert_to_server 开始 ...', raw_func.__name__, server_name)
            servicer = Servicer(args[0])
            serve(servicer, max_send_message_length=256, max_receive_message_length=256)
            # ret = raw_func(*args, **kwargs)
            # print('convert_to_server 结束 ...')
            return 


        return wrapper


    return _convert_to_server










