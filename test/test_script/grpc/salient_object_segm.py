from redpy.grpc import CommonClient

if __name__ == '__main__':
    import cv2
    img = cv2.imread('/share/wangqixun/workspace/github_project/redpy/test/data/beauty.jpg')
    
    client = CommonClient(
        ip='10.4.200.43',
        port='30126',
        max_send_message_length=100, 
        max_receive_message_length=100,
    )
    res = client.run([img])
    print(res.shape)





