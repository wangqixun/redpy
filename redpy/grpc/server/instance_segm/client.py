import grpc
import instance_segm_pb2 as pb2
import instance_segm_pb2_grpc as pb2_grpc
import pickle
import numpy as np
import pickle
import cv2

__all__ = [
    'Client'
]

def run(img, ip, port, max_send_message_length=100, max_receive_message_length=100):
    # 连接 rpc 服务器
    channel = grpc.insecure_channel(
        f'{ip}:{port}',
        options=[
            ('grpc.max_send_message_length', max_send_message_length * 1024 * 1024),
            ('grpc.max_receive_message_length', max_receive_message_length * 1024 * 1024)
        ]
    )
    # 调用 rpc 服务
    stub = pb2_grpc.InstanceSegmStub(channel)
    img_bgr_bytes = pickle.dumps(img)
    input_grpc = pb2.ISOneImgRequest(
        img_bgr_bytes=img_bgr_bytes,
    )

    response = stub.infer_one_img(input_grpc)
    result_bytes = pickle.loads(response.result_bytes)
    print(result_bytes)


class MMDetSOD(object):
    def __init__(self, topk, classes):
        self.top_k = topk
        self.CLASSES = classes

    def iou_batch(self, a, b):
        pass 

    def merge_contours(self, contours):
        res = []
        for contour in contours:
            contour = contour.reshape([-1, 2]).tolist()
            res += contour
        res = np.array(res).reshape([-1, 2])
        return res

    def iou_of_maskA_and_maskB(self, maskA_list, maskB_list):
        res = np.zeros([len(maskA_list), len(maskB_list)])
        for a in range(len(maskA_list)):
            for b in range(len(maskB_list)):
                maskA, maskB = maskA_list[a], maskB_list[b]
                iou = (maskA & maskB).sum() / (1e-8 + (maskA | maskB).sum())
                res[a, b] = iou
        return res
    
    def distance_of_contoursA_and_contoursB(self, contoursA, contoursB):
        res = np.zeros([len(contoursA), len(contoursB)])
        for a in range(len(contoursA)):
            for b in range(len(contoursB)):
                contours_A, contours_B = contoursA[a][:, None, :], contoursB[b][None, :, :]
                dist_AB = np.sum((contours_A - contours_B) ** 2, axis=2) ** 0.5
                dist_min = np.min(dist_AB)
                res[a, b] = dist_min
        return res

    def mmdet_result_post_processing(self, res_mmdet, conf=0.7, keep_label_list=None, bbox=None):
    
        for idx, label_name in enumerate(self.CLASSES):
            if label_name not in keep_label_list:
                res_mmdet[0][idx] = np.empty([0, 5], dtype=np.float32)
                res_mmdet[1][idx] = []
            else:
                keep_mask = res_mmdet[0][idx][:, 4] >= conf
                res_mmdet[0][idx] = res_mmdet[0][idx][keep_mask]
                res_mmdet[1][idx] = [res_mmdet[1][idx][idx_km] for idx_km, m in enumerate(keep_mask) if m]

        person_areas = [seg.sum() for seg in res_mmdet[1][0]]
        if len(person_areas) == 0:
            # salient_person_idx = -1
            res_det = [np.empty([0, 5], dtype=np.float32) for idx in range(len(self.CLASSES))]
            res_seg = [[] for idx in range(len(self.CLASSES))]
            return (res_det, res_seg)
        if bbox is None:
            # print("bbox is None")
            salient_person_idx  = np.argmax(person_areas)
        else:
            person_bbox_ious    = self.iou_batch(np.array([bbox]),res_mmdet[0][0][:,:4])
            salient_person_idx  = np.argmax(person_bbox_ious)
            # print("bbox is",bbox,"chosen is",res_mmdet[0][0][salient_person_idx])
        contours_person = [
            cv2.findContours((res_mmdet[1][0][idx]*255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
             for idx in range(len(res_mmdet[1][0]))
        ]
        contours_person = [self.merge_contours(contours) for contours in contours_person]
        masks_person = [mask for mask in res_mmdet[1][0]]

        for idx_class, label_name in enumerate(self.CLASSES):
            if label_name in ['person']:
                keep_list = [salient_person_idx]
                res_mmdet[0][idx_class] = res_mmdet[0][idx_class][keep_list]
                res_mmdet[1][idx_class] = [res_mmdet[1][idx_class][m] for m in keep_list]
                continue
            if len(res_mmdet[0][idx_class]) == 0:
                continue
            contours_idx_class = [
                cv2.findContours((res_mmdet[1][idx_class][idx]*255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
                for idx in range(len(res_mmdet[1][idx_class]))
            ]
            contours_idx_class = [self.merge_contours(contours) for contours in contours_idx_class]
            masks_idx_class = [mask for mask in res_mmdet[1][idx_class]]

            distance_list = self.distance_of_contoursA_and_contoursB(contours_idx_class, contours_person)
            distance_keep_list = [idx for idx in range(len(distance_list)) if (np.argmin(distance_list[idx])==salient_person_idx) and (np.min(distance_list[idx]) <= (min(self.raw_img_shape[:2])/10))]

            iou_list = self.iou_of_maskA_and_maskB(masks_idx_class, masks_person)
            iou_keep_list = [idx for idx in range(len(iou_list)) if iou_list[idx][salient_person_idx] > 0]

            keep_list = [idx for idx in range(len(res_mmdet[0][idx_class])) if idx in distance_keep_list or idx in iou_keep_list]
            
            if len(keep_list) == 0:
                res_mmdet[0][idx_class] = np.empty([0, 5], dtype=np.float32)
            else:
                res_mmdet[0][idx_class] = res_mmdet[0][idx_class][keep_list][:self.top_k]
            res_mmdet[1][idx_class] = [res_mmdet[1][idx_class][m] for m in keep_list][:self.top_k]

        return res_mmdet


class Client():
    def __init__(self, ip, port, max_send_message_length=100, max_receive_message_length=100):
        classes = [
            'person', 'bicycle', 'motorcycle', 'frisbee', 'snowboard', 'sports ball', 
            'baseball bat', 'skateboard', 'tennis racket',
        ]

        # 连接 rpc 服务器
        channel = grpc.insecure_channel(
            f'{ip}:{port}',
            options=[
                ('grpc.max_send_message_length', max_send_message_length * 1024 * 1024),
                ('grpc.max_receive_message_length', max_receive_message_length * 1024 * 1024)
            ]
        )
        stub = pb2_grpc.InstanceSegmStub(channel)

        self.stub = stub
        self.classes = classes
        self.sod = MMDetSOD(5, classes)


    def run(self, img, sod=False, sod_cfg={'conf':0.4}):
        # 调用 rpc 服务
        img_bgr_bytes = pickle.dumps(img)
        input_grpc = pb2.ISOneImgRequest(
            img_bgr_bytes=img_bgr_bytes,
        )
        response = self.stub.infer_one_img(input_grpc)
        result = pickle.loads(response.result_bytes)

        if sod:
            result = self.sod.mmdet_result_post_processing(
                res_mmdet=result,
                conf=sod_cfg.get('conf', 0.4),
                keep_label_list=self.classes,
                bbox=None,
            )
        return result

        

if __name__ == '__main__':
    img = cv2.imread('/share/wangqixun/workspace/github_project/redpy/test/data/beauty.jpg')
    client = Client(
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







