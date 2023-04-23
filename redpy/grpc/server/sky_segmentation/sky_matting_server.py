import cv2
import numpy as np
# pip install git+https://github.com/huggingface/transformers.git
import queue
import onnxruntime
import copy

from redpy.grpc.server.common.server import convert_to_server

queue_maxsize = 2

class SkyMatting(object):
    
    def __init__(self, queue_maxsize=1, device_list=["cuda:0"]):
        ckp_file = '/share/wangqixun/workspace/github_project/skyseg_u2net/output/onnx/sky_matting_v3.onnx'
        # ckp_file = '/share/yaoyifan/workspace/Sky-Segmentation-and-Post-processing/v4_ft.onnx'

        self.queue = queue.Queue(maxsize=queue_maxsize)
        for i in range(queue_maxsize):
            onnx_session = onnxruntime.InferenceSession(ckp_file, providers=onnxruntime.get_available_providers())
            self.queue.put(onnx_session)        

    @convert_to_server("SkyMattingServer", 30125, queue_maxsize)
    def run_inference(self, img, input_size=[320, 320]):
        '''
            img: bgr, HW3
        '''
        onnx_session = self.queue.get(True, timeout=10)

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
        input_name = onnx_session.get_inputs()[0].name
        output_name = onnx_session.get_outputs()[0].name
        onnx_result = onnx_session.run([output_name], {input_name: x})

        # Post process
        onnx_result = np.array(onnx_result).squeeze()
        min_value = np.min(onnx_result)
        max_value = np.max(onnx_result)
        # onnx_result = (onnx_result - min_value) / (max_value - min_value)
        onnx_result *= 255
        onnx_result = onnx_result.astype('uint8')

        onnx_result = cv2.resize(onnx_result, (img.shape[1], img.shape[0]))

        if onnx_session is not None:
            self.queue.put(onnx_session)        

        return onnx_result



if __name__ == '__main__':
    
    sm = SkyMatting(queue_maxsize=queue_maxsize)

    img = cv2.imread('/share/wangqixun/workspace/github_project/redpy/test/data/skiing.png')
    res = sm.run_inference(img)
    cv2.imwrite('/share/wangqixun/workspace/tmp/0000.jpg', res)





