import cv2
import torch
import numpy as np
from PIL import Image
import queue
from redpy.grpc.server.common.server import convert_to_server
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



class Depth(object):
    
    def __init__(self, queue_maxsize=2, device_list=["cuda:0"]):

        self.queue = queue.Queue(maxsize=queue_maxsize)
        self.device_list = device_list
        for i in range(queue_maxsize):
            model, transform = init_model('dpt_large')
            device = torch.device(i % len(device_list))
            self.queue.put([model, transform, device])        


    @convert_to_server("DepthServer", 30301, 2)
    def infer(self, img):
        '''
            img: bgr channel
        '''
        model, transform, device = self.queue.get(True, timeout=10)

        img_input = transform({"image": img[..., ::-1] / 255.0})["image"]

        # compute
        with torch.no_grad():
            sample = torch.from_numpy(img_input).to(device).unsqueeze(0)
            prediction = model.forward(sample)
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

        if model is not None:
            self.queue.put([model, transform, device])        

        return img_vis


if __name__ == '__main__':
    
    # load model
    model = Depth(device_list=["cuda:0"])
    
    # load image
    # image = np.zeros([512, 512, 3], dtype=np.uint8) + 128
    image = cv2.imread('/share/wangqixun/workspace/github_project/multimodal-intelligence/wqx_business/pixar/data/person/2.jpg')
    
    # infer
    res = model.infer(image)
    print(res)




