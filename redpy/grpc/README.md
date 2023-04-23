# 当前服务统计

若无特殊说明，涉及到图像的输入默认使用BGR通道。由于需求量和机器原因，部分服务（translate、instance segm等）已经下线，需要的话再联系

服务名称 | IP | Port 
:-------------------------:|:-------------------------:|:-------------------------:
[多人脸属性-arcface](redpy/grpc/server/arcface/client.py)| 10.4.200.42|30390
[深度-MiDaS](redpy/grpc/server/depth_estimation)|10.4.200.42|30301
[BLIP2](redpy/grpc/server/blip2/client.py)|10.4.200.42|30302
[天空-matting](redpy/grpc/server/sky_segmentation)|10.4.200.42|30125
[肢体-mmpose](redpy/grpc/server/mmpose)|10.4.200.42|12358
[图像打标-tagger](redpy/grpc/server/tagger/client.py)|10.4.200.42|30126




