# 超事件的核心代码（时序结构滤波）

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

class TSF(nn.Module):

    def __init__(self, N=3):
        super(TSF, self).__init__()

        self.N = float(N)  # 每个卷积核中柯西分布的个数
        self.Ni = int(N)   # 每个卷积核的通道数

        # create parameteres for center and delta of this super event
	
        self.center = nn.Parameter(torch.FloatTensor(N)) # 卷积核的参数
        self.delta = nn.Parameter(torch.FloatTensor(N))
        self.gamma = nn.Parameter(torch.FloatTensor(N))

        # init them around 0

        self.center.data.normal_(0,0.5)
        self.delta.data.normal_(0,0.01)
        self.gamma.data.normal_(0, 0.0001)


    def get_filters(self, delta, gamma, center, length, time):
        """
            delta (batch,) in [-1, 1]
            center (batch,) in [-1, 1]
            gamma (batch,) in [-1, 1]
            length (batch,) of ints
        """

        # scale to length of videos
	# 下面都是6是因为batch_size为2，每一个batch为3
        centers = (length - 1) * (center + 1) / 2.0   #[6]
        deltas = length * (1.0 - torch.abs(delta)) # [6]

        gammas = torch.exp(1.5 - 2.0 * torch.abs(gamma)) # [6]
        
        a = Variable(torch.zeros(self.Ni)) 
        a = a.cuda()
        
        # stride and center
        a = deltas[:, None] * a[None, :]  # None可以理解为按某一行或一列依次运行 [6,3]，将卷积核的通道数拓展到Ni
        a = centers[:, None] + a # [2*3,3]

        b = Variable(torch.arange(0, time))  # [0,1,2,3,......31]  [32,]
        b = b.cuda()
        '''
	b = [0,1,2,.............31]
	
	a = [20.1715,20.1715,20.1715]
	    [6.1892,6.1892,6.1892]
	    [19.0191,19.0191,19.0191]
	    
	    
	    
	b-a[:,:,None] = [[0,1,2,...31]-[20.1715]
	                [0,1,2,...31]-[20.1715]
			[0,1,2,...31]-[20.1715]]
			[[0,1,2,...31]-[6.1892]
			 [0,1,2,...31]-[6.1892]
			 [0,1,2,...31]-[6.1892]]
			[[0,1,2,...31]-[19.0191]
			 [0,1,2,...31]-[19.0191]
			 [0,1,2,...31]-[19.0191]]
	
	'''
        f = b - a[:, :, None]   # [6,3,32]  
        f = f / gammas[:, None, None]
        
        f = f ** 2.0
        f += 1.0
        f = np.pi * gammas[:, None, None] * f
        f = 1.0/f
        f = f/(torch.sum(f, dim=2) + 1e-6)[:,:,None]   # (6,3,32),对第二个维度聚合，相当于对所有的数值进行了归一化

        f = f[:,0,:].contiguous()     # 由于第一个维度是一样的，因此移除了第一个维度 (6,32)

        f = f.view(-1, self.Ni, time)  # (2,3,32)  第二个3代表有3种卷积核
        
        return f

    def forward(self, inp):
        video, length = inp  # (B,C,T,H,W)，length为传入的mask向量长度和时间维度一致
        batch, channels, time = video.squeeze(3).squeeze(3).size()  # 
        # vid is (B x C x T)
        vid = video.view(batch*channels, time, 1).unsqueeze(2) # (B*C,T,1,1)--
        # f is (B x N x T)
        f = self.get_filters(torch.tanh(self.delta).repeat(batch), torch.tanh(self.gamma).repeat(batch), torch.tanh(self.center.repeat(batch)), length.view(batch,1).repeat(1,self.Ni).view(-1), time)
        # repeat over channels
        f = f.unsqueeze(1).repeat(1, channels, 1, 1)  # (2,1024,3,32)-->(B,C,N,T)
        f = f.view(batch*channels, self.Ni, time)  # (B*C,N,T)

        # o is (B x C x N)
        o = torch.bmm(f, vid.squeeze(2))  # (B*C,N,T)X(B*C,T,1) = (B*C,N,1)  
        del f
        del vid
        o = o.view(batch, channels*self.Ni)#.unsqueeze(3).unsqueeze(3)  也就是每个特征图用三种不同的全局特征加权
	return o



