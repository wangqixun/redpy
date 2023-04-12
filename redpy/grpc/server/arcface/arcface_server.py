import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
# from insightface.data import get_image as ins_get_image

import queue
from redpy.grpc.server.common.server import convert_to_server
import time
import numpy as np
import pickle


maxsize = 2


class ArcfaceServer():
    def __init__(self, maxsize=2):
        self.queue = queue.Queue(maxsize)
        for i in range(maxsize):
            app = FaceAnalysis('antelopev2', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
            app.prepare(ctx_id=0, det_size=(640, 640))
            self.queue.put(app)


    @convert_to_server(server_name='ArcfaceServer', port=30390, max_workers=maxsize)
    def infer(self, img):
        app = self.queue.get(True, timeout=10)
        faces = app.get(img)
        if app is not None:
            self.queue.put(app)
        print(f"{len(faces)} faces")

        res = []
        for idx_person in range(len(faces)):
            face = faces[idx_person]
            face_info = {}
            for k,v in face.items():
                if isinstance(v, np.ndarray):
                    face_info[k] = v.tolist()
                else:
                    face_info[k] = v
            res.append(face_info)
        return res


if __name__ == "__main__":
    service = ArcfaceServer(maxsize=maxsize)
    img = cv2.imread('/share/wangqixun/workspace/github_project/redpy/test/data/touxiang2.jpeg')
    res = service.infer(img)
    print(res)
    # print(res)



