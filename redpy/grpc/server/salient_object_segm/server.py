from concurrent import futures
import sys
sys.path.append('/share/wangqixun/workspace/github_project/DIS/IS-Net')
import torch.nn.functional as F
from torchvision.transforms.functional import normalize
import torch
from models import *
import numpy as np
import cv2
import queue

from redpy.grpc.server.common.server import convert_to_server


# 实现 proto 文件中定义的 Servicer
class SalientObjectSegm(object):
    def __init__(self, maxsize=4):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_size = [1024, 1024]

        self.queue = queue.Queue(maxsize)
        for i in range(maxsize):
            model_path="/share/wangqixun/workspace/github_project/DIS/IS-Net/output/v5_fiou/checkpoint_epoch073_loss0.20714_fiou0.9069.pth.tar"  # the model path
            # model_path = "/share/wangqixun/workspace/github_project/DIS/saved_models/isnet-general-use.pth"  # the model path
            net = ISNetDIS()
            net.to(self.device)
            checkpoint = torch.load(model_path, map_location="cpu")
            net.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()})
            # net.load_state_dict(torch.load(model_path))
            net.eval()
            self.queue.put(net)


    @convert_to_server('SalientObjectSegmServer', port=30126, max_workers=2)
    def infer(self, img):
        '''
            img: bgr channel
        '''
        model = self.queue.get(True, timeout=10)

        im = img[..., ::-1] # RGB
        im_shp = im.shape[0:2]
        im_tensor = torch.tensor(np.copy(im), dtype=torch.float32).permute(2,0,1)
        im_tensor = F.upsample(torch.unsqueeze(im_tensor,0), self.input_size, mode="bilinear").type(torch.uint8)
        image = im_tensor / 255.
        image = normalize(image[0],[0.5,0.5,0.5],[1.0,1.0,1.0])[None]

        image = image.to(self.device)
        with torch.no_grad():
            result = model(image)
        result=torch.squeeze(F.upsample(result[0][0], im_shp, mode='bilinear'), 0)
        ma = torch.max(result)
        mi = torch.min(result)
        # result = (result-mi)/(ma-mi)
        # if ma < 0.99:
        #     result *= 0
        result = result.permute(1,2,0).cpu().data.numpy()
        result = result * 255
        result = result.astype(np.uint8)

        if model is not None:
            self.queue.put(model)

        return result


if __name__ == '__main__':
    salient_object_segm = SalientObjectSegm(maxsize=2)
    img = cv2.imread('/share/wangqixun/workspace/github_project/redpy/test/data/beauty.jpg')
    res = salient_object_segm.infer(img)
    cv2.imwrite('/share/wangqixun/workspace/github_project/redpy/test_img.jpg', res)

