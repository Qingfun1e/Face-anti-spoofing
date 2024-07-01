import torch
import torch.nn as nn
import torch.nn.functional as F
from models.swin_unet import ModifiedUNet
import numpy as np


class TripletLoss(nn.Module):
    def __init__(self, margin=0.2):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, f_anchor, f_positive, f_negative):
        f_anchor, f_positive, f_negative = renorm(f_anchor), renorm(f_positive), renorm(f_negative)
        # print(f_anchor.shape)
        b = f_anchor.size(0)  # sample count batch size = 12
        f_anchor = f_anchor.view(b, -1)
        f_positive = f_positive.view(b, -1)
        f_negative = f_negative.view(b, -1)  # 拉平  (n,1,k) [(12,2048)]
        # print(f_anchor.shape)
        with torch.no_grad():
            idx = hard_samples_mining(f_anchor, f_positive, f_negative, self.margin)
            # print(idx)

        d_ap = torch.norm(f_anchor[idx] - f_positive[idx], dim=1)
        d_an = torch.norm(f_anchor[idx] - f_negative[idx], dim=1)
        # print("d_an:{}".format(d_an))
        # print("d_ap:{}".format(d_ap))
        # q_all = d_ap + d_an
        # q_ap = torch.pow(d_ap, 2)
        # q_an = torch.pow(d_an, 2)
        # q_ap = torch.exp(q_ap)
        # q_an = torch.exp(1 - q_an)

        # print(q_ap)

        # lax = torch.pow(q_ap, 2) * q_ap + torch.pow(1 - q_an, 2) * (1 - q_an)

        # d_pn = torch.norm(f_positive[idx] - f_negative[idx], dim=1)
        # lt2 = torch.clamp((d_ap - 0.1 * self.margin), 0).mean()
        # print(d_ap)
        # print(d_an)
        # print(d_pn)
        # return torch.clamp(d_ap - 1 / 2 * (d_an + d_pn) + self.margin, 0).mean() + 0.1 * lt2

        # lz = torch.clamp(1 - torch.pow(d_an, 2), 0)
        lz = torch.clamp(-(1 - d_an) ** 2 * torch.log(d_an), 0)
        # y = np.maximum(-(1 - x) ** 2 * np.log(x), 0)
        # print('lz:{}'.format(lz))
        lz_loss = lz.mean()
        rl_loss = torch.clamp(d_ap - d_an + self.margin, 0).mean()

        # print('lz_loss:{}'.format(lz_loss))
        # print('rl_loss:{}'.format(rl_loss))

        return rl_loss + lz_loss


def hard_samples_mining(f_anchor, f_positive, f_negative, margin):
    d_ap = torch.norm(f_anchor - f_positive, dim=1)  # 横向 范式
    d_an = torch.norm(f_anchor - f_negative, dim=1)
    # jia
    # d_pn = torch.norm(f_positive - f_negative, dim=1)

    # idx = d_ap - d_an < margin
    # print(d_an-d_ap)
    idx = d_an - d_ap < margin
    # idx = (d_ap - 1 / 2 * (d_pn + d_an)) < margin
    # print(idx)
    return idx


def renorm(x):
    # w.renorm(2,0,1e-5).mul(le5) 对w进行归一化 。w.renorm中前两个2，
    # 0是代表在对w进行在第0维度的L2范数操作得到归一化结果。
    # 1e-5是代表maxnorm ，将大于1e-5的乘以1e5，使得最终归一化到1
    return x.renorm(2, 0, 1e-5).mul(1e5)  # torch.renorm(x,2,0,1e-5).mul(1e5)


class TotalLoss(nn.Module):
    def __init__(self, margin=0.5):
        super(TotalLoss, self).__init__()
        self.margin = margin
        self.trip = TripletLoss(margin)
        self.reg = nn.MSELoss()
        self.cla = nn.CrossEntropyLoss()

    def forward(self, regression, classification, feat, labels):
        regression_anchor, regression_positive, regression_negative = regression
        b, c, _, _ = regression_anchor.size()
        classification_anchor, classification_positive, classification_negative = classification

        feat_anchor, feat_positive, feat_negative = feat
        # print(regression_anchor.device)
        # print(regression_positive.device)
        # print(regression_negative.device)
        # print(labels.device)
        # labels 0 正正负
        # print(regression_anchor)

        reg_loss = self.reg(regression_negative[labels == 1],
                            torch.zeros_like(regression_negative[labels == 1]).cuda()) + \
                   self.reg(
                       regression_anchor[labels == 0], torch.zeros_like(regression_anchor[labels == 0]).cuda()) + \
                   self.reg(
                       regression_positive[labels == 0], torch.zeros_like(regression_positive[labels == 0]).cuda())

        cla_loss = self.cla(classification_anchor[labels == 0],
                            torch.tensor([1] * classification_anchor[labels == 0].size(0), dtype=torch.long).cuda()
                            ) + \
                   self.cla(classification_anchor[labels == 1],
                            torch.tensor([0] * classification_anchor[labels == 1].size(0), dtype=torch.long).cuda()
                            ) + \
                   self.cla(classification_positive[labels == 0],
                            torch.tensor([1] * classification_positive[labels == 0].size(0), dtype=torch.long).cuda()
                            ) + \
                   self.cla(classification_positive[labels == 1],
                            torch.tensor([0] * classification_positive[labels == 1].size(0), dtype=torch.long).cuda()
                            ) + \
                   self.cla(classification_negative[labels == 0],
                            torch.tensor([0] * classification_negative[labels == 0].size(0), dtype=torch.long).cuda()
                            ) + \
                   self.cla(classification_negative[labels == 1],
                            torch.tensor([1] * classification_negative[labels == 1].size(0), dtype=torch.long).cuda()
                            )

        trip_loss = sum([self.trip(a, b, c) for a, b, c in zip(feat_anchor, feat_positive, feat_negative)])
        # print('reg_loss:{}'.format(reg_loss))
        # print('cla_loss:{}'.format(cla_loss))
        # print('trip_loss:{}'.format(trip_loss))
        # print(e_loss)
        # lrs = reg_loss
        return 5*reg_loss + 5*cla_loss + trip_loss


if __name__ == "__main__":
    regression = [torch.randn(1, 3, 24, 24), torch.randn(1, 3, 24, 24), torch.randn(1, 3, 24, 24)]
    classifictions = [torch.randn(1, 2), torch.randn(1, 2), torch.randn(1, 2)]
    feat = [[torch.randn(12, 12, 1, 1), torch.randn(1, 1, 1, 1)],
            [torch.randn(12, 12, 1, 1), torch.randn(1, 1, 1, 1)],
            [torch.randn(12, 12, 1, 1), torch.randn(1, 1, 1, 1)]]
    labels = torch.tensor([1], dtype=torch.long)
    tl_loss = TotalLoss()
    res = tl_loss(regression, classifictions, feat, labels)
