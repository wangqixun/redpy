import grpc
import depth_estimation_pb2 as pb2
import depth_estimation_pb2_grpc as pb2_grpc
from concurrent import futures
import time
import pickle
from redpy.utils_redpy.logger_utils import setup_logger
logger = setup_logger(
    '/share/wangqixun/workspace/github_project/redpy/redpy/grpc/server/depth_estimation/log.log',
    name='DepthEstimationServer'
)


import sys
sys.path.append('/share/wangqixun/workspace/github_project/MiDaS')
from midas.dpt_depth import DPTDepthModel
from midas.midas_net import MidasNet
from midas.midas_net_custom import MidasNet_small
from midas.transforms import Resize, NormalizeImage, PrepareForNet
from midas import transforms

import numpy as np
import cv2
import torch
from torchvision.transforms import Compose


def init_model(model_type="midas_v21_small", model_path_dict=None):
    '''model_type:
    midas_v21_small
    midas_v21
    dpt_hybrid
    dpt_large
    '''
    if model_path_dict is None:
        model_path_dict = {
            'midas_v21_small':'/share/wangqixun/workspace/github_project/MiDaS/model_dl/model-small-70d6b9c8.pt',
            'midas_v21':'/share/wangqixun/workspace/github_project/MiDaS/model_dl/midas_v21-f6b98070.pt',
            'dpt_hybrid':'/share/wangqixun/workspace/github_project/MiDaS/model_dl/dpt_hybrid-midas-501f0c75.pt',
            'dpt_large':'/share/wangqixun/workspace/github_project/MiDaS/model_dl/dpt_large-midas-2f21e586.pt',
        }
    model_path = model_path_dict[model_type]

    logger.info("initialize model")

    # select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("device: %s" % device)

    # load network
    if model_type == "dpt_large": # DPT-Large
        model = DPTDepthModel(
            path=model_path,
            backbone="vitl16_384",
            non_negative=True,
        )
        net_w, net_h = 384, 384
        resize_mode = "minimal"
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    elif model_type == "dpt_hybrid": #DPT-Hybrid
        model = DPTDepthModel(
            path=model_path,
            backbone="vitb_rn50_384",
            non_negative=True,
        )
        net_w, net_h = 384, 384
        resize_mode="minimal"
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    elif model_type == "midas_v21":
        model = MidasNet(model_path, non_negative=True)
        net_w, net_h = 384, 384
        resize_mode="upper_bound"
        normalization = NormalizeImage(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
    elif model_type == "midas_v21_small":
        model = MidasNet_small(model_path, features=64, backbone="efficientnet_lite3", exportable=True, non_negative=True, blocks={'expand': True})
        net_w, net_h = 256, 256
        resize_mode="upper_bound"
        normalization = NormalizeImage(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
    else:
        logger.info(f"model_type '{model_type}' not implemented, use: --model_type large")
        assert False
    
    transform = Compose(
        [
            Resize(
                net_w,
                net_h,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method=resize_mode,
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            normalization,
            PrepareForNet(),
        ]
    )

    model.eval()
    model.to(device)

    return model, transform


# 实现 proto 文件中定义的 Servicer
class DepthEstimation(pb2_grpc.DepthEstimationServicer):
    def __init__(self):
        super(pb2_grpc.DepthEstimationServicer).__init__()
        model, transform = init_model('dpt_large')
        self.model = model
        self.transform = transform
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def infer(self, img):
        '''
            img: bgr channel
        '''
        img_input = self.transform({"image": img[..., ::-1] / 255.0})["image"]

        # compute
        with torch.no_grad():
            sample = torch.from_numpy(img_input).to(self.device).unsqueeze(0)
            prediction = self.model.forward(sample)
            prediction = (
                torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=img.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                )
                .squeeze()
                .cpu()
                .numpy()
            )
    
        img_vis = prediction
        img_vis = (img_vis - img_vis.min()) / (img_vis.max() - img_vis.min() + 1e-8)
        img_vis = (img_vis * 255).astype(np.uint8)
        return img_vis

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
            logger.info(f'   [{t3-t2:.4f}] depth estimation infer finish.')
            res_bytes = pickle.dumps(res_img)
            output = pb2.DEOneImgReply(
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
    pb2_grpc.add_DepthEstimationServicer_to_server(DepthEstimation(), server)
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
        port=31002,
        max_send_message_length=256,  # 256m
        max_receive_message_length=256,  # 256m
    )

