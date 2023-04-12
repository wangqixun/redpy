import cv2
import torch
import numpy as np
from PIL import Image
# pip install git+https://github.com/huggingface/transformers.git
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import queue

from redpy.grpc.server.common.server import convert_to_server

class blip2(object):
    
    def __init__(self, queue_maxsize=1, device_list=["cuda:0"]):
        model_path = "/share/wangqixun/workspace/github_project/model_dl/blip2-opt-2.7b"

        self.queue = queue.Queue(maxsize=queue_maxsize)
        self.device_list = device_list
        for i in range(queue_maxsize):

            device = torch.device(i % len(device_list))
            processor = Blip2Processor.from_pretrained(model_path)
            model = Blip2ForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.float16)
            model.to(device)

            # warm up
            image = np.zeros([256, 256, 3], dtype=np.uint8)
            inputs = processor(images=image, return_tensors="pt").to(device, torch.float16)
            _ = model.generate(**inputs)

            self.queue.put([processor, model, device])        

    
    @convert_to_server("Blip2Server", 30302, 1)
    def infer(self, image):
        
        if isinstance(image, str):
            image = Image.open(image)
        elif isinstance(image, np.ndarray):
            # BGR format
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        processor, model, device = self.queue.get(True, timeout=10)

        inputs = processor(images=image, return_tensors="pt").to(device, torch.float16)
        generated_ids = model.generate(**inputs)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        if processor is not None and model is not None:
            self.queue.put([processor, model, device])        

        return generated_text


if __name__ == '__main__':
    
    # load model
    blip2_model = blip2(queue_maxsize=1, device_list=["cuda:0"])
    
    # load image
    # image = np.zeros([512, 512, 3], dtype=np.uint8) + 128
    image = cv2.imread('/share/wangqixun/workspace/github_project/multimodal-intelligence/wqx_business/pixar/data/person/2.jpg')
    
    # infer
    caption = blip2_model.infer(image)
    print(caption)