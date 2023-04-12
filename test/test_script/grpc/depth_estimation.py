from redpy.grpc import DepthEstimationClient
import cv2

if __name__ == '__main__':
    img = cv2.imread('/share/wangqixun/workspace/github_project/redpy/test/data/beauty.jpg')
    client = DepthEstimationClient(
        ip='10.4.200.43',
        port='31002',
        max_send_message_length=100, 
        max_receive_message_length=100,
    )
    res = client.run(img)
    cv2.imwrite('/share/wangqixun/workspace/github_project/redpy/test/data/demo_output/depth_estimation.jpg', res)


