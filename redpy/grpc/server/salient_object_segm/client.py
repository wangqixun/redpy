from redpy.grpc import CommonClient


def infer(img):
    client = CommonClient(
        ip='10.4.200.43',
        port='30126',
        max_send_message_length=100, 
        max_receive_message_length=100,
    )
    res = client.run([img])
    print(res.shape)
    return res


if __name__ == '__main__':
    import cv2
    img = cv2.imread('/share/wangqixun/workspace/github_project/redpy/test/data/beauty.jpg')
    
    # from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED, ALL_COMPLETED
    # import time
    # t1 = time.time()
    # with ThreadPoolExecutor(max_workers=10) as t:
    #     all_task = [t.submit(infer, img) for i in range(50)]
    #     wait(all_task, return_when=ALL_COMPLETED)
    # t2 = time.time()
    # total = 50
    # cost = t2 - t1
    # print("cost: {}, total: {}, qps: {}".format(cost, total, (total/cost)))

    infer(img)





