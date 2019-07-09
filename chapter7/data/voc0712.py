"""VOC Dataset Classes

Original author: Francisco Massa
https://github.com/fmassa/vision/blob/voc_dataset/torchvision/datasets/voc.py

Updated by: Ellis Brown, Max deGroot
"""
"""
Copyright (c) 2017 Max deGroot, Ellis Brown
Released under the MIT license
https://github.com/amdegroot/ssd.pytorch
Updated by: Takuya Mouri
"""
from .config import HOME
import os.path as osp
import os
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np
import json
from pathlib import Path

VOC_CLASSES = (  # always index 0
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')

character_list = [str(i) for i in range(10)]

# handbook
# note: if you used our download scripts, this should be right
#VOC_ROOT = osp.join(HOME, "data/VOCdevkit/")
# 現在のディレクトリを取得
dir_cur = osp.dirname(__file__)
# データセットVOCのディレクトリを取得
dir_voc = osp.join(dir_cur, "..", "VOCdevkit")
# データセットVOCの絶対パスを設定
VOC_ROOT = osp.abspath(dir_voc)
# handbook

class VOCAnnotationTransform(object):
    """Transforms a VOC annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes

    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VOC's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """

    def __init__(self, root='input', transform=None):
        self.root = root

    def __call__(self, target, width, height):
        res = []
        pts = ['xmin', 'ymin', 'xmax', 'ymax']
        with open(target, 'r') as fo:
            for obj in json.load(fo)['bboxes']:
                label = obj['char']
                if label in character_list:
                    bndbox = []
                    for i, pt in enumerate(pts):
                        cur_pt = obj[pt]
                        cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
                        bndbox.append(cur_pt)
                    bndbox.append(int(label))
                    res += [bndbox]
        return res      # [[xmin, ymin, xmax, ymax, label_ind], ... ]


class VOCDetection(data.Dataset):
    """VOC Detection Dataset Object

    input is image, target is annotation

    Arguments:
        root (string): filepath to VOCdevkit folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'VOC2007')
    """

    def __init__(self, root,
                 image_sets = ['train',  'test'],
                 transform=None, target_transform=VOCAnnotationTransform(),
                 dataset_name='Fuji'):
        self.root = root
        self.image_set = image_sets
        self.transform = transform
        self.target_transform = target_transform
        self.name = dataset_name
        if self.image_set=='train':
            root_images_path = os.path.join(self.root, 'train_images')
            root_ans_path = os.path.join(self.root, 'train_anns')
            ans_path_list = sorted(Path(root_ans_path).glob('*.json'))
            images_path_list = [Path(str(path).replace('_anns', '_images')).with_suffix('.jpg')
                                for path in ans_path_list]
            self.ans_path_list =  ans_path_list
            self.image_path_list = images_path_list
        else:
            im ='a'

        self.num = len(self.image_path_list)


    def __getitem__(self, index):
        im, gt, h, w = self.pull_item(index)
        return im, gt

    def __len__(self):
        return self.num

    def pull_item(self, index):

        target = self.ans_path_list[index]
        img = cv2.imread(str(self.image_path_list[index]))
        height, width, channels = img.shape

        if self.target_transform is not None:
            target = self.target_transform(target, width, height)

        if self.transform is not None:
            target = np.array(target)
            img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
            # to rgb
            # cv2のchannelsはbgrなのでrgbの順番に変更
            img = img[:, :, (2, 1, 0)]
            # img = img.transpose(2, 0, 1)
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        # 画像の次元の順番をHWCからCHWに変更
        return torch.from_numpy(img).permute(2, 0, 1), target, height, width
        # return torch.from_numpy(img), target, height, width

    def pull_image(self, index):
        img_id = self.image_path_list[index]
        return cv2.imread(str(img_id))

    def pull_anno(self, index):
        img_id = self.ids[index]
        anno = ET.parse(self._annopath % img_id).getroot()
        gt = self.target_transform(anno, 1, 1)
        return img_id[1], gt

    def pull_tensor(self, index):
        return torch.Tensor(self.pull_image(index)).unsqueeze_(0)
