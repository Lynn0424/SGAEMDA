import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import *
# nm = 495
# nd = 383
# nc = 5430
#
# miRNAnumber = np.genfromtxt(r'miRNA number.txt', dtype=str, delimiter='\t')
# diseasenumber = np.genfromtxt(r'disease number.txt', dtype=str, delimiter='\t')
#
# DS = np.loadtxt(r'd-d.txt', delimiter=',')
# MS = np.loadtxt(r'm-m.txt', delimiter=',')
# MS = normalized(MS)
# DS = normalized(DS)
# DS = torch.from_numpy(DS).float()
# MS = torch.from_numpy(MS).float()
# A = np.zeros((nm, nd), dtype=float)
# ConnectData = np.loadtxt(r'D:\WritingCode\SAEMDA-main\known miRNA-disease associations.txt', dtype=int) - 1
#
# for i in range(nc):
#     A[ConnectData[i, 0], ConnectData[i, 1]] = 1
#
# # 取所有未关联样本对的坐标用data0_index保存起来
# data0_index = np.argwhere(A == 0)  # (184155,2)
#
# A = torch.from_numpy(A).float()
#
# A = A.cuda()
# DS = DS.cuda()
# MS = MS.cuda()

# 定义卷积层
class GraphConv(nn.Module):
    def __init__(self, in_dim, out_dim, drop=0.5, bias=False, activation=None):
        super(GraphConv, self).__init__()
        self.dropout = nn.Dropout(drop)
        self.activation = activation
        self.w = nn.Linear(in_dim, out_dim, bias=bias)
        nn.init.xavier_uniform_(self.w.weight)
        self.bias = bias
        if self.bias:
            nn.init.zeros_(self.w.bias)

    def forward(self, adj, x):
        x = self.dropout(x)
        x = adj.mm(x)
        x = self.w(x)
        if self.activation:
            return self.activation(x)
        else:
            return x


# 定义自编码器
class LP(nn.Module):
    def __init__(self, hid_dim, out_dim, bias=False):
        super(LP, self).__init__()
        self.res1 = GraphConv(out_dim, hid_dim, bias=bias, activation=F.relu)
        self.res2 = GraphConv(hid_dim, hid_dim, bias=bias, activation=torch.tanh)
        self.res3 = GraphConv(hid_dim, hid_dim, bias=bias, activation=F.relu)
        self.res4 = GraphConv(hid_dim, out_dim, bias=bias, activation=torch.sigmoid)

    def forward(self, adj, x):
        z = self.res2(adj, self.res1(adj, x))   # z表示低维嵌入
        x_ = self.res4(adj, self.res3(adj, z)) # x_表示重构输出
        return x_, z


# 定义堆叠自编码器
class SGAE(nn.Module):
    def __init__(self,hid_dim1,hid_dim2,hid_dim3,out_dim):
        super(SGAE, self).__init__()
        self.gae1 = LP(hid_dim1, out_dim)
        self.gae2 = LP(hid_dim2, hid_dim1)
        self.gae3 = LP(hid_dim3,hid_dim2)

    def forward(self,adj, y0):
        y1, z1 = self.gae1(adj,y0)
        # print('y1:',y1.shape,'z1:',z1.shape)
        y2, z2 = self.gae2(adj,z1)
        # print('y2',y2.shape,'z2', z2.shape)
        y3, z3 = self.gae3(adj,z2)
        # print('y3',y3.shape,'z3', z3.shape)
        return z1, z2, z3, y1, y2, y3

    def my_mse_loss(self,adj,y0):
        z1, z2, z3, y1, y2, y3 = self.forward(adj,y0)
        return torch.mean(torch.pow((y0 - y1), 2)) + torch.mean(torch.pow((z1 - y2), 2)) + torch.mean(torch.pow((z2 - y3), 2))

# class MLP(nn.Module):
#     def __init__(self):
#         super(MLP, self).__init__()
#         self.sequential = nn.Sequential(
#             nn.Linear(256,128),
#             nn.Tanh(),
#             nn.Linear(128,1),
#             nn.Sigmoid()
#         )
#     def forward(self,x):
#         x = self.sequential(x)
#         return x

