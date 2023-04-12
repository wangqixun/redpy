from redpy.grpc import DiffusionClient
import cv2


if __name__ == '__main__':
    # text = '美丽的动漫女性肖像，亚洲人，娃娃脸，穿着黑色连帽衫，干净的脸，枯死的眼睛，棕色的头发，对称的面部，艺术站，人物概念艺术，迪斯尼海报，2d插图，逼真的脸，高细节，电影，8k，8k高清，平面设计，--ar 3:4-uplight'
    client = DiffusionClient(
        ip='10.4.200.42',
        port='30128',
        max_send_message_length=100, 
        max_receive_message_length=100,
    )

    prompt = "<tlshu>, bodybuilding"
    neg_prompt = "text"

    input_dict = dict(
        prompt=prompt,
        neg_prompt=neg_prompt,
    )
    res = client.run(
        input_dict,
    )
    cv2.imwrite('./test_img.jpg', res)

