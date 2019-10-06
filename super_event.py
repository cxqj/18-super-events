import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import temporal_structure_filter as tsf


class SuperEvent(nn.Module):
    def __init__(self, classes=65):
        super(SuperEvent, self).__init__()

        self.classes = classes
        self.dropout = nn.Dropout(0.7)
        self.add_module('d', self.dropout)

        # 定义两个超事件
        self.super_event = tsf.TSF(3)  # 3可以理解为通道数，其实超事件可以理解为深度可分离卷积，对每个通道用一个卷积核做卷积，每个卷积核的参数分布满足柯西分布
        self.add_module('sup', self.super_event)
        self.super_event2 = tsf.TSF(3)
        self.add_module('sup2', self.super_event2)


        # we have 2xD*3
        # we want to learn a per-class weighting
        # to take 2xD*3 to D*3
        self.cls_wts = nn.Parameter(torch.Tensor(classes))  # soft_attention权重
        
        # 超事件的权重
        self.sup_mat = nn.Parameter(torch.Tensor(1, classes, 1024*3))  # class为滤波器的个数，其中每个滤波器有三个通道，作用于输入的每个通道
        stdv = 1./np.sqrt(1024+1024)
        self.sup_mat.data.uniform_(-stdv, stdv)

        # 最终的输出和其卷积核参数的分布
        self.per_frame = nn.Conv3d(1024, classes, (1,1,1))
        self.per_frame.weight.data.uniform_(-stdv, stdv)
        self.per_frame.bias.data.uniform_(-stdv, stdv)
        self.add_module('pf', self.per_frame)
        
    def forward(self, inp):
        inp[0] = self.dropout(inp[0])
        val = False
        dim = 1
        if inp[0].size()[0] == 1:
            val = True
            dim = 0

        #print inp[0].size()
        # 最终的超事件表示是两种超事件的concat
        # 每个超事件的维度为(B,C*3)
        super_event = self.dropout(torch.stack([self.super_event(inp).squeeze(), self.super_event2(inp).squeeze()], dim=dim))  # (B,2,C*3)
        if val:
            super_event = super_event.unsqueeze(0)
        # we have B x 2 x D*3
        # we want B x C x D*3

        #print super_event.size()
        # now we have C x 2 matrix
        cls_wts = torch.stack([torch.sigmoid(self.cls_wts), 1-torch.sigmoid(self.cls_wts)], dim=1)

        # now we do a bmm to get B x C x D*3
        #print cls_wts.expand(inp[0].size()[0], -1, -1).size(), super_event.size()
        # (2,65,2) X (2,2,3072)---->(2,65,3072)
        super_event = torch.bmm(cls_wts.expand(inp[0].size()[0], -1, -1), super_event)    # 再用类别权重对每个通道加权
        del cls_wts
        print super_event.size()
        # apply the super-event weights
        super_event = torch.sum(self.sup_mat * super_event, dim=2)  # (2,65)  将所有的通道聚合为一个分类概率
        #super_event = self.sup_mat(super_event.view(-1, 1024)).view(-1, self.classes)
        
        super_event = super_event.unsqueeze(2).unsqueeze(3).unsqueeze(4)  # (2,65,1,1,1)

        cls = self.per_frame(inp[0])  # (2,65,32,1,1)
        return super_event+cls    # (2,65,32,1,1)



def get_super_event_model(gpu, classes=65):
    model = SuperEvent(classes)
    return model.cuda()

