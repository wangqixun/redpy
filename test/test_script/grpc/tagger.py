import cv2
import time
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED, ALL_COMPLETED
# from rich import print

from redpy.grpc import CommonClient

def infer():
    image = cv2.imread('/share/wangqixun/workspace/github_project/multimodal-intelligence/wqx_business/pixar/data/person/2.jpg')
    client = CommonClient(
        ip='10.4.200.42',
        port='30126',
        max_send_message_length=100, 
        max_receive_message_length=100,
    )
    res = client.run([image])
    # cv2.imwrite('./test_img.jpg', res)

    print(res)
    # ['1girl', 'solo', 'looking at viewer', 'brown hair', 'shirt', 'brown eyes', 'jewelry', 'white shirt', 
    # 'earrings', 'outdoors', 'sky', 'day', 'cloud', 'bag', 'blue sky', 'lips', 'sunglasses', 'grass', 
    # 'eyewear on head', 'mountain', 'realistic', 'selfie']


def qps():
    t1 = time.time()
    with ThreadPoolExecutor(max_workers=10) as t:
        all_task = [t.submit(infer, ) for i in range(50)]
        wait(all_task, return_when=ALL_COMPLETED)
    t2 = time.time()

    total = 50
    cost = t2 - t1
    print("cost: {}, total: {}, qps: {}".format(cost, total, (total/cost)))


def compare():
    img_list = [
        '/share/wangqixun/workspace/github_project/redpy/test/data/beauty.jpg',
        '/share/wangqixun/workspace/github_project/redpy/test/data/datou_1.png',
        '/share/wangqixun/workspace/github_project/redpy/test/data/2girl.jpeg',
        '/share/wangqixun/workspace/github_project/redpy/test/data/girl_1.jpg',
        '/share/wangqixun/workspace/github_project/redpy/test/data/anime_1.png',
        '/share/wangqixun/workspace/github_project/redpy/test/data/anime_2.png',
    ]
    client_tagger = CommonClient(
        ip='10.4.200.42',
        port='30126',
        max_send_message_length=100, 
        max_receive_message_length=100,
    )
    client_blip2 = CommonClient(
        ip='10.4.200.42',
        port='30302',
        max_send_message_length=100, 
        max_receive_message_length=100,
    )

    for img_file in img_list:
        image = cv2.imread(img_file)
        res_tagger = client_tagger.run([image])
        res_blip2 = client_blip2.run([image])

        print(img_file)
        print(res_blip2)
        print(', '.join(res_tagger))


if __name__ == '__main__':
    # img = cv2.imread('/share/wangqixun/workspace/github_project/redpy/test/data/beauty.jpg')
    # infer()
    # qps()
    compare()


