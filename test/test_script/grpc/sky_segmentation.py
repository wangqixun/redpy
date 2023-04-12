from redpy.grpc import SkySegmentationClient
import cv2

if __name__ == '__main__':
    img = cv2.imread('/share/wangqixun/workspace/github_project/redpy/test/data/tennis.png')
    client = SkySegmentationClient(
        ip='10.4.200.42',
        port='30125',
        max_send_message_length=100, 
        max_receive_message_length=100,
    )
    res = client.run(img)
    cv2.imwrite('/share/wangqixun/workspace/github_project/redpy/test/data/demo_output/sky_segmentation.jpg', res)

