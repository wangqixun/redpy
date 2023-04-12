import hashlib
import os
import time
# from rich import print
# import json
import ffmpeg
import base64
import numpy as np
import pycocotools.mask as mask_util
from .json_utils import load_json, write_json


__all__ = [
    'run_in_cmd', 'getfilemd5', 'get_cur_time', 'get_basename_without_suffix', 'get_video_info',
    'encode_img_to_str', 'mask2rle', 'rle2mask'
]

def run_in_cmd(cmd, print_info=True):
    res = os.system(cmd)
    if print_info:
        if res:
            print('RUN ERROR', cmd)
        else:
            print('RUN SUCCESS', cmd)


def getfilemd5(filename):
    if not os.path.isfile(filename):
        return
    myhash = hashlib.md5()
    f = open(filename, 'rb')
    while True:
        b = f.read(8096)
        if not b:
            break
        myhash.update(b)
    f.close()
    return myhash.hexdigest()


def get_cur_time():
    t_str = time.strftime('%Y%m%d-%H%M%S', time.localtime(time.time()))
    return t_str


def get_basename_without_suffix(file_path):
    basename = os.path.basename(file_path)
    res = basename.split('.')[:-1]
    res = '.'.join(res)
    return res


def get_video_info(video_path, logger=None):
    basename = get_basename_without_suffix(video_path)
    cache_file = os.path.join(os.path.dirname(video_path), f"{basename}_info.json")

    vinfo = None
    if os.path.exists(cache_file):
        if logger is not None:
            logger.debug(f'video info cache found: {cache_file}, reading...')
        vinfo = load_json(cache_file)
    if vinfo is None:
        vinfo = ffmpeg.probe(video_path)
        # vinfo = ffmpeg_tool.get_video_info(video_path)
        if logger is not None:
            logger.debug(f'save video_info cache to: {cache_file}, writing...')
        if vinfo is not None:
            write_json(vinfo, cache_file)
    return vinfo


def encode_img_to_str(img):
    img_str = base64.b64encode(img).decode('ascii')
    return img_str


def decode_img_from_str(img_str, shape):
    img_decode = base64.b64decode(img_str.encode('ascii')) # 解base64编码，得图片的二进制
    try:
        img_array = np.frombuffer(img_decode, np.uint8).reshape(shape)
    except ValueError:
        img_array = np.frombuffer(img_decode, np.float).reshape(shape)
    return img_array


def mask2rle(mask):
    '''
        mask=np.array(
                [
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 1, 0, 0, 1, 0],
                    [0, 0, 1, 1, 1, 1, 1, 0],
                    [0, 0, 1, 1, 1, 1, 1, 0],
                    [0, 0, 1, 1, 1, 1, 1, 0],
                    [0, 0, 1, 0, 0, 0, 1, 0],
                    [0, 0, 1, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0]
                ]
            )
    '''
    rle = mask_util.encode(np.array(mask[:, :, None], order='F', dtype="uint8"))[0]
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def rle2mask(coco_image_segmentation):
    '''
        coco_image_segmentation
    '''
    uncompressed_rle = coco_image_segmentation
    compressed_rle = mask_util.frPyObjects(uncompressed_rle, uncompressed_rle.get('size')[0], uncompressed_rle.get('size')[1])

    # get binary mask
    polygon_mask = mask_util.decode(compressed_rle)
    return polygon_mask


singleMask2rle = mask2rle


def nms_cpu(dects, threshhold):
    #dects 二维数组（n_samples， 5） 即 x1,y1,x2,y2,score
    #threshhold IOU阈值
    x1 = dects[:, 0]  # pred bbox top_x #[100. 250. 220. 100. 230. 220.]
    y1 = dects[:, 1]  # pred bbox top_y #[100. 250. 220. 100. 240. 230.] 
    x2 = dects[:, 2]  # pred bbox bottom_x #[210. 420. 320. 210. 325. 315.]
    y2 = dects[:, 3]  # pred bbox bottom_y #[210. 420. 330. 210. 330. 340.]
    scores = dects[:, 4]  # pred bbox cls score #[0.72 0.8  0.92 0.72 0.81 0.9 ]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1) #各个框的面积
    index = scores.argsort()[::-1] #[2 5 4 1 3 0]#分数从大到小排列的index,[::-1]是列表头和尾颠倒一下。
    #argsort()用法，表示对数据进行从小到大进行排序，返回数据的索引值。scores.argsort()--> [0 3 1 4 5 2]
    #即 分数从小到大排列为(scores[0] scores[3] scores[1] scores[4] scores[5] scores[2])及(.72 .72 .8 .81 .9 .91)
    # 对应从大到小的索引  index[  2   5    4     1    3   0  ]记住是取出索引，scores列表没变。

    keep = []#符合条件索引的index，keep用于存放NMS后剩余的方框， # keep保留的是索引值，不是具体的分数。

    # index会剔除遍历过的方框，和合并过的方框。
    while index.size > 0:
        i = index[0]# 取出第一个索引号
        keep.append(i)

        # 计算交集的左上角和右下角
        #np.maximum(X, Y) 用于逐元素比较两个array的大小。就是分数最大的x1值与剩下的按序排列分数对应X1值的挨个对比，较大者。
        #x1[i]为取出分数最大的索引位置对应的x1值，x1[index[1:]]为后续从大到小分数索引位置对应的x1值
        xx1 = np.maximum(x1[i], x1[index[1:]]) #[220. 230. 250. 220. 220.]
        yy1 = np.maximum(y1[i], y1[index[1:]]) #[230. 240. 250. 220. 220.]
        xx2 = np.minimum(x2[i], x2[index[1:]]) #[315. 320. 320. 210. 210.]
        yy2 = np.minimum(y2[i], y2[index[1:]]) #[330. 330. 330. 210. 210.]
        # 如果两个方框相交，X22-X11和Y22-Y11是正的。
        # 如果两个方框不相交，X22-X11和Y22-Y11是负的，不相交的W和H设为0


        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)

        # 计算重叠面积就是上面说的交集面积。
        inter = w * h    #[9696. 8281. 5751.    0.    0.]  #不相交因为W和H都是0 ，不相交面积为0

        # IOU公式（交并比）。
        # 得出来的ious是一个列表，里面拥有当前方框和其他所有方框的IOU结果。
        ious = inter / (areas[i] + areas[index[1:]] - inter) #[0.79664777 0.70984056 0.16573009 0.         0.        ]


        # 合并重叠度最大的方框，也是合并ious中值大于thresh的方框
        # 合并的操作就是把他们去掉，合并这些方框只保留下分数最高的。
        # 经过排序当前操作的方框就是分数最高的，所以剔除其他和当前重叠度最高的方框
        # 这里np.where(ious<=thresh)[0]是一个固定写法。
        inds = np.where(ious <= threshhold)[0]  #[2 3 4] #ious中第2、3、4位置的小于IOU阈值（不包含分数最高的，ious中5个数），也就是index中的第3、4、5位置（包含最高的，index中6个数）

        # 把留下来框在进行NMS操作
        # 这边留下的框是去除当前操作的框，和当前操作的框重叠度大于thresh的框
        # 每一次都会先去除当前操作框，所以索引的列表就会向前移动移位，要还原就+1，向后移动一位
        index = index[inds + 1]  #[1 3 0]  ##index=[2 5 4 1 3 0]，对应index第3、4、5位置  变为-->index=[1 3 0]

    return keep




if __name__ == '__main__':
    # f = '/share/wangqixun/xiuchang/output/211107_one/live-montage-8734440506_1636042719084_1636042759002_v2.mp4'
    # md5 = getfilemd5(f)
    # print(md5)

    # 每个类别都有很多重叠的候选框。
    # 最后，可以通过NMS算法进行筛选，最终得到了分类器认为置信度最高的框作为最后的预测框。
    boxes = np.array([[100, 100, 210, 210, 0.72],
                    [250, 250, 420, 420, 0.8],
                    [220, 220, 320, 330, 0.92],
                    [100, 100, 210, 210, 0.72],
                    [230, 240, 325, 330, 0.81],
                    [220, 230, 315, 340, 0.9]])
    keep = nms_cpu(boxes, 0.7)
    print(keep)