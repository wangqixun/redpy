import math
import cv2
import torch
import numpy as np
from itertools import product
from mmengine.registry import init_default_scope

from mmpose.apis import inference_topdown
from mmpose.apis import init_model as init_pose_estimator
from mmpose.evaluation.functional import nms
from mmpose.structures import merge_data_samples
from mmdet.apis import inference_detector, init_detector
from redpy.grpc.server.common.server import convert_to_server

# openpose format
limb_seq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10],
            [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17],
            [1, 16], [16, 18]]

colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255,
                                                     0], [170, 255, 0],
          [85, 255, 0], [0, 255, 0],
          [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255],
          [0, 85, 255], [0, 0, 255], [85, 0, 255], [170, 0,
                                                    255], [255, 0, 255],
          [255, 0, 170], [255, 0, 85]]

stickwidth = 4
num_openpose_kpt = 18
num_link = len(limb_seq)

class MmposeEstimator:

    def __init__(self):

        det_config = "/share/xingchuan/data/models/mmpose/rtmdet_nano_320-8xb32_coco-person.py"
        det_checkpoint = "/share/xingchuan/data/models/mmpose/rtmdet_nano_8xb32-100e_coco-obj365-person-05d8511e.pth"

        pose_config = "/share/xingchuan/data/models/mmpose/rtmpose-m_8xb256-420e_coco-256x192.py"
        pose_checkpoint = "/share/xingchuan/data/models/mmpose/rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-256x192-63eb25f7_20230126.pth"

        self.bbox_thr = 0.4
        self.nms_thr = 0.3
        self.kpt_thr=0.4

        self.detector = init_detector(det_config, det_checkpoint)
        self.pose_estimator = init_pose_estimator(
            pose_config,
            pose_checkpoint,
            cfg_options=dict(model=dict(test_cfg=dict(output_heatmaps=False))))
    
    
    @convert_to_server(server_name='mmpose', port=12358, max_workers=10)
    def infer(self, image, input_format="BGR", output_format="BGR"):
        """Visualize predicted keypoints of one image in openpose format."""

        if input_format=="BGR":
            image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        try:
            # predict bbox
            init_default_scope(self.detector.cfg.get('default_scope', 'mmdet'))
            det_result = inference_detector(self.detector, image)
            pred_instance = det_result.pred_instances.cpu().numpy()
            bboxes = np.concatenate(
            (pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
            bboxes = bboxes[np.logical_and(pred_instance.labels == 0,
                                    pred_instance.scores > self.bbox_thr)]
            bboxes = bboxes[nms(bboxes, self.nms_thr), :4]

            # predict keypoints
            pose_results = inference_topdown(self.pose_estimator, image, bboxes)
            data_samples = merge_data_samples(pose_results)

            # concatenate scores and keypoints
            keypoints = np.concatenate(
                (data_samples.pred_instances.keypoints,
                data_samples.pred_instances.keypoint_scores.reshape(-1, 17, 1)),
                axis=-1)

            # compute neck joint
            neck = (keypoints[:, 5] + keypoints[:, 6]) / 2
            for index in range(len(keypoints)):
                if keypoints[index, 5, 2] < self.kpt_thr or keypoints[index, 6, 2] < self.kpt_thr:
                    neck[index, 2] = 0

            # 17 keypoints to 18 keypoints
            new_keypoints = np.insert(keypoints[:, ], 17, neck, axis=1)

            # mmpose format to openpose format
            openpose_idx = [15, 14, 17, 16, 2, 6, 3, 7, 4, 8, 12, 9, 13, 10, 1]
            mmpose_idx = [1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17]
            new_keypoints[:, openpose_idx, :] = new_keypoints[:, mmpose_idx, :]

            # black background
            black_img = np.zeros_like(image)

            num_instance = new_keypoints.shape[0]

            # draw keypoints
            for i, j in product(range(num_instance), range(num_openpose_kpt)):
                x, y, conf = new_keypoints[i][j]
                if conf > self.kpt_thr:
                    cv2.circle(black_img, (int(x), int(y)), 4, colors[j], thickness=-1)

            # draw links
            cur_black_img = black_img.copy()
            for i, link_idx in product(range(num_instance), range(num_link)):
                conf = new_keypoints[i][np.array(limb_seq[link_idx]) - 1, 2]
                if np.sum(conf > self.kpt_thr) == 2:
                    Y = new_keypoints[i][np.array(limb_seq[link_idx]) - 1, 0]
                    X = new_keypoints[i][np.array(limb_seq[link_idx]) - 1, 1]
                    mX = np.mean(X)
                    mY = np.mean(Y)
                    length = ((X[0] - X[1])**2 + (Y[0] - Y[1])**2)**0.5
                    angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
                    polygon = cv2.ellipse2Poly(
                        (int(mY), int(mX)), (int(length / 2), stickwidth), int(angle),
                        0, 360, 1)
                    cv2.fillConvexPoly(cur_black_img, polygon, colors[link_idx])
            black_img = cv2.addWeighted(black_img, 0.4, cur_black_img, 0.6, 0)
        except Exception as e:
            print("errors")
            black_img = np.zeros_like(image).astype(np.uint8)

        if output_format == "BGR":
            black_img = cv2.cvtColor(black_img,cv2.COLOR_RGB2BGR)
        return black_img


if __name__ == "__main__":
    img = cv2.imread("/share/xingchuan/code/onlinedemos/open-pose/test.jpg")
    
    mmpose = MmposeEstimator()

    res = mmpose.infer(img)
    cv2.imwrite("1.jpg", res)






        
    
        









