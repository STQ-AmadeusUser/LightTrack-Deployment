import torch
import numpy as np
import torch.nn as nn

from .backbone.backbone import ChildNet_FCN
from .neck.neck import BN_adj, PW_Corr_adj
from .head.head import head_subnet


class LightTrackM_Subnet(nn.Module):
    def __init__(self, search_size=256, template_size=128, stride=16):
        super(LightTrackM_Subnet, self).__init__()
        self.search_size = search_size
        self.template_size = template_size
        self.stride = stride
        self.score_size = round(self.search_size / self.stride)
        self.num_kernel = round(self.template_size / self.stride) ** 2
        self.criterion = nn.BCEWithLogitsLoss()
        self.retrain = False
        self.init_arch()

    def init_arch(self):
        self.features = ChildNet_FCN()
        self.neck = BN_adj(num_channel=96)
        self.feature_fusor = PW_Corr_adj(num_kernel=64, adj_channel=128)
        self.head = head_subnet()

    def forward(self, template, search, label, reg_target, reg_weight):
        '''backbone_index: which layer's feature to use'''
        zf = self.features(template)
        xf = self.features(search)
        # Batch Normalization before Corr
        zf, xf = self.neck(zf, xf)
        # Point-wise Correlation
        feat_dict = self.feature_fusor(zf, xf)
        # supernet head
        oup = self.head(feat_dict)
        # compute loss
        reg_loss = self.add_iouloss(oup['reg'], reg_target, reg_weight)
        cls_loss = self._weighted_BCE(oup['cls'], label)
        return cls_loss, reg_loss

    def template(self, z):
        self.zf = self.features(z)

    def track(self, x):
        # supernet backbone
        xf = self.features(x)
        # BN before Pointwise Corr
        zf, xf = self.neck(self.zf, xf)
        # Point-wise Correlation
        feat_dict = self.feature_fusor(zf, xf)
        # supernet head
        oup = self.head(feat_dict)
        return oup['cls'], oup['reg']

    def grids(self):
        """
        each element of feature map on input search image
        :return: H*W*2 (position for each element)
        """
        sz = self.score_size
        print('grids size=', sz)

        sz_x = sz // 2
        sz_y = sz // 2

        x, y = np.meshgrid(np.arange(0, sz) - np.floor(float(sz_x)),
                           np.arange(0, sz) - np.floor(float(sz_y)))

        self.grid_to_search = {}
        self.grid_to_search_x = x * self.stride + self.search_size // 2
        self.grid_to_search_y = y * self.stride + self.search_size // 2

        self.grid_to_search_x = torch.Tensor(self.grid_to_search_x).unsqueeze(0).unsqueeze(0).cuda()
        self.grid_to_search_y = torch.Tensor(self.grid_to_search_y).unsqueeze(0).unsqueeze(0).cuda()

        self.grid_to_search_x = self.grid_to_search_x.repeat(self.batch, 1, 1, 1)
        self.grid_to_search_y = self.grid_to_search_y.repeat(self.batch, 1, 1, 1)

    def _weighted_BCE(self, pred, label, mode='all'):
        pred = pred.view(-1)
        label = label.view(-1)
        if mode == 'pos' or mode == 'all':
            pos = label.data.eq(1).nonzero().squeeze()
            loss_pos = self._cls_loss(pred, label, pos)
        if mode == 'neg' or mode == 'all':
            neg = label.data.eq(0).nonzero().squeeze()
            loss_neg = self._cls_loss(pred, label, neg)
        # return
        if mode == 'pos':
            return loss_pos
        elif mode == 'neg':
            return loss_neg
        elif mode == 'all':
            return loss_pos * 0.5 + loss_neg * 0.5

    def _cls_loss(self, pred, label, select):
        if len(select.size()) == 0: return 0
        pred = torch.index_select(pred, 0, select)
        label = torch.index_select(label, 0, select)
        return self.criterion(pred, label)  # the same as tf version

    def _IOULoss(self, pred, target, weight=None):
        pred_left = pred[:, 0]
        pred_top = pred[:, 1]
        pred_right = pred[:, 2]
        pred_bottom = pred[:, 3]

        target_left = target[:, 0]
        target_top = target[:, 1]
        target_right = target[:, 2]
        target_bottom = target[:, 3]

        target_area = (target_left + target_right) * (target_top + target_bottom)
        pred_area = (pred_left + pred_right) * (pred_top + pred_bottom)

        w_intersect = torch.min(pred_left, target_left) + torch.min(pred_right, target_right)
        h_intersect = torch.min(pred_bottom, target_bottom) + torch.min(pred_top, target_top)

        area_intersect = w_intersect * h_intersect
        area_union = target_area + pred_area - area_intersect

        losses = -torch.log((area_intersect + 1.0) / (area_union + 1.0))

        if weight is not None and weight.sum() > 0:
            return (losses * weight).sum() / weight.sum()
        else:
            assert losses.numel() != 0
            return losses.mean()

    def add_iouloss(self, bbox_pred, reg_target, reg_weight, iou_mode='iou'):
        """

        :param bbox_pred:
        :param reg_target:
        :param reg_weight:
        :param grid_x:  used to get real target bbox
        :param grid_y:  used to get real target bbox
        :return:
        """
        assert (iou_mode == 'iou' or iou_mode == 'diou')
        bbox_pred_flatten = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
        reg_target_flatten = reg_target.reshape(-1, 4)
        reg_weight_flatten = reg_weight.reshape(-1)
        pos_inds = torch.nonzero(reg_weight_flatten > 0).squeeze(1)

        bbox_pred_flatten = bbox_pred_flatten[pos_inds]
        reg_target_flatten = reg_target_flatten[pos_inds]
        if iou_mode == 'iou':
            loss = self._IOULoss(bbox_pred_flatten, reg_target_flatten)
        elif iou_mode == 'diou':
            loss = self._DIoU_Loss(bbox_pred_flatten, reg_target_flatten)
        else:
            raise ValueError('iou_mode should be iou or diou')
        return loss

    def pred_to_image(self, bbox_pred):
        self.grid_to_search_x = self.grid_to_search_x.to(bbox_pred.device)
        self.grid_to_search_y = self.grid_to_search_y.to(bbox_pred.device)

        pred_x1 = self.grid_to_search_x - bbox_pred[:, 0, ...].unsqueeze(1)  # 17*17
        pred_y1 = self.grid_to_search_y - bbox_pred[:, 1, ...].unsqueeze(1)  # 17*17
        pred_x2 = self.grid_to_search_x + bbox_pred[:, 2, ...].unsqueeze(1)  # 17*17
        pred_y2 = self.grid_to_search_y + bbox_pred[:, 3, ...].unsqueeze(1)  # 17*17

        pred = [pred_x1, pred_y1, pred_x2, pred_y2]

        pred = torch.cat(pred, dim=1)

        return pred
