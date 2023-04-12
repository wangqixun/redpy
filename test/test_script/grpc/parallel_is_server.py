from redpy.utils_redpy.parallel_utils import parallel_wrapper
import time
import cv2
from redpy.grpc import InstanceSegmClient


@parallel_wrapper(50, 10000)
# @parallel_wrapper(8, 10000, cache='cache')
def say(say_list, ):
    for info in say_list:
        # bgr通道顺序图像
        img = cv2.imread('/share/wangqixun/workspace/github_project/redpy/test/data/beauty.jpg')
        # inference
        client = InstanceSegmClient(
            ip='10.4.200.42',
            port='30123',
            max_send_message_length=100, 
            max_receive_message_length=100,
        )
        res = client.run(
            img,
            sod=True,  # 主体检测
            sod_cfg=dict(  # 主体检测参数
                conf=0.4
            ),
        )
        print(info)


if __name__ == '__main__':
    # 初始化client

    t = time.time()
    say_list = [_ for _ in range(100)]
    say(say_list)
    print(time.time() - t)





