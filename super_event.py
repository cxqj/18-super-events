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
        self.super_event = tsf.TSF(3)
        self.add_module('sup', self.super_event)
        self.super_event2 = tsf.TSF(3)
        self.add_module('sup2', self.super_event2)


        # we have 2xD*3
        # we want to learn a per-class weighting
        # to take 2xD*3 to D*3
        self.cls_wts = nn.Parameter(torch.Tensor(classes))  # (65)
        
        # 超事件的权重
        self.sup_mat = nn.Parameter(torch.Tensor(1, classes, 1024*3))  #(1,65,1024*3)
        stdv = 1./np.sqrt(1024+1024)
        self.sup_mat.data.uniform_(-stdv, stdv)

        # 获取输入特征对应的帧级分类结果，最后和输出的超事件表达相结合
        self.per_frame = nn.Conv3d(1024, classes, (1,1,1))  # 1024-->65
        self.per_frame.weight.data.uniform_(-stdv, stdv)
        self.per_frame.bias.data.uniform_(-stdv, stdv)
        self.add_module('pf', self.per_frame)
        
    def forward(self, inp):  # inp[0]:(2,1024,32,1,1)  inp[1]:[32,32] inp[1]记录了特征的实际长度
        inp[0] = self.dropout(inp[0])
        
        # 判断当前是否为验证集
        val = False
        dim = 1
        if inp[0].size()[0] == 1:
            val = True
            dim = 0

        # 每个超事件的输出为(2,3072),两个超事件的输出拼接后为(2,2,3072)
        super_event = self.dropout(torch.stack([self.super_event(inp).squeeze(), self.super_event2(inp).squeeze()], dim=dim))  # (2,2,3072)
        if val:
            super_event = super_event.unsqueeze(0)
            
        # we have B x 2 x D*3
        # we want B x C x D*3
        #接下来需要获取每个类别对应的超事件，就是通过cls_wts矩阵与超事件的输出相乘即可得到
        # now we have C x 2 matrix
        cls_wts = torch.stack([torch.sigmoid(self.cls_wts), 1-torch.sigmoid(self.cls_wts)], dim=1)

        # now we do a bmm to get B x C x D*3
        #print cls_wts.expand(inp[0].size()[0], -1, -1).size(), super_event.size()
        # (2,65,2) X (2,2,3072)---->(2,65,3072)
        super_event = torch.bmm(cls_wts.expand(inp[0].size()[0], -1, -1), super_event)    # 获取每个类别对应的超事件
        del cls_wts
        print super_event.size()
        
        # apply the super-event weights 注意这里是直接相乘不是矩阵乘
        # (2,65,3072)*(2,65,3072)-->(2,65,3072)-->(2,65)
        super_event = torch.sum(self.sup_mat * super_event, dim=2)  # (2,65)  将所有的通道聚合为一个分类概率
        #super_event = self.sup_mat(super_event.view(-1, 1024)).view(-1, self.classes)
        
        super_event = super_event.unsqueeze(2).unsqueeze(3).unsqueeze(4)  # (2,65,1,1,1)

        cls = self.per_frame(inp[0])  # (2,65,32,1,1)
        return super_event+cls    # (2,65,32,1,1)



def get_super_event_model(gpu, classes=65):
    model = SuperEvent(classes)
    return model.cuda()



