from redpy.grpc import InstanceSegmClient
import cv2

if __name__ == '__main__':
    img = cv2.imread('/share/wangqixun/workspace/github_project/redpy/test/data/skiing.png')
    client = InstanceSegmClient(
        ip='10.4.200.42',
        port='30123',
        max_send_message_length=100, 
        max_receive_message_length=100,
    )

    res = client.run(
        img,
        sod=False,  # 主体检测
        sod_cfg=dict(  # 主体检测参数
            conf=0.4
        ),
    )







