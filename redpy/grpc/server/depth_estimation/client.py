import numpy as np
import cv2
import pickle
from redpy.grpc import CommonClient

        

if __name__ == '__main__':
    img = cv2.imread('/share/wangqixun/workspace/github_project/redpy/test/data/beauty.jpg')
    client = CommonClient(
        ip='10.4.200.42',
        port='30301',
        max_send_message_length=100, 
        max_receive_message_length=100,
    )
    res = client.run([img])
    print(res)







