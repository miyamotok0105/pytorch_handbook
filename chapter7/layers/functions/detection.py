"""
Copyright (c) 2017 Max deGroot, Ellis Brown
Released under the MIT license
https://github.com/amdegroot/ssd.pytorch
Updated by: Takuya Mouri
"""
import torch
from torch.autograd import Function
from ..box_utils import decode, nms
from data import voc as cfg


class Detect(Function):
    """At test time, Detect is the final layer of SSD.  Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations.
    """
    # PyTorch1.5.0 support new-style autograd function
    #def __init__(self, num_classes, bkg_label, top_k, conf_thresh, nms_thresh):
    #    self.num_classes = num_classes
    #    self.background_label = bkg_label
    #    self.top_k = top_k
    #    # Parameters used in nms.
    #    self.nms_thresh = nms_thresh
    #    if nms_thresh <= 0:
    #        raise ValueError('nms_threshold must be non negative.')
    #    self.conf_thresh = conf_thresh
    #    self.variance = cfg['variance']

    #def forward(self, loc_data, conf_data, prior_data):
    @staticmethod
    def forward(self, num_classes, bkg_label, top_k, conf_thresh, nms_thresh, loc_data, conf_data, prior_data):
        self.num_classes = num_classes
        self.background_label = bkg_label
        self.top_k = top_k
        # Parameters used in nms.
        self.nms_thresh = nms_thresh
        if nms_thresh <= 0:
            raise ValueError('nms_threshold must be non negative.')
        self.conf_thresh = conf_thresh
        self.variance = cfg['variance']
    # PyTorch1.5.0 support new-style autograd function
        """
        Args:
            loc_data: (tensor) Loc preds from loc layers
                Shape: [batch,num_priors*4]
            conf_data: (tensor) Shape: Conf preds from conf layers
                Shape: [batch*num_priors,num_classes]
            prior_data: (tensor) Prior boxes and variances from priorbox layers
                Shape: [1,num_priors,4]
        """
        num = loc_data.size(0)  # batch size
        num_priors = prior_data.size(0)
        # [バッチサイズN,クラス数21,トップ200件,確信度+位置]のゼロリストを作成
        output = torch.zeros(num, self.num_classes, self.top_k, 5)
        # 確信度を[バッチサイズN,クラス数,ボックス数]の順番に変更
        conf_preds = conf_data.view(num, num_priors,
                                    self.num_classes).transpose(2, 1)

        # Decode predictions into bboxes.
        for i in range(num):
            decoded_boxes = decode(loc_data[i], prior_data, self.variance)
            # For each class, perform nms
            conf_scores = conf_preds[i].clone()

            for cl in range(1, self.num_classes):
                # 確信度の閾値を使ってボックスを削除
                c_mask = conf_scores[cl].gt(self.conf_thresh)
                scores = conf_scores[cl][c_mask]
                # handbook
                #if scores.dim() == 0:
                if scores.size(0) == 0:
                # handbook
                    continue
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
                # ボックスのデコード処理
                boxes = decoded_boxes[l_mask].view(-1, 4)
                # idx of highest scoring and non-overlapping boxes per class
                # boxesからNMSで重複するボックスを削除
                ids, count = nms(boxes, scores, self.nms_thresh, self.top_k)
                output[i, cl, :count] = \
                    torch.cat((scores[ids[:count]].unsqueeze(1),
                               boxes[ids[:count]]), 1)
        flt = output.contiguous().view(num, -1, 5)
        _, idx = flt[:, :, 0].sort(1, descending=True)
        _, rank = idx.sort(1)
        flt[(rank < self.top_k).unsqueeze(-1).expand_as(flt)].fill_(0)
        return output
