from redpy.grpc import CommonClient
import cv2 

client = CommonClient(
    ip='10.4.200.42',
    port='12358',
    max_send_message_length=100, 
    max_receive_message_length=100
    )

img = cv2.imread("/share/xingchuan/code/onlinedemos/open-pose/test.jpg")
res = client.run([img])
cv2.imwrite("1.jpg", res)
