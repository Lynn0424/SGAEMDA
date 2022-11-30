# 开发时间  2022/6/13 17:31
# 开发时间  2022/5/18 16:22
from __future__ import division, print_function, absolute_import
import argparse
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import *
from sklearn.utils import shuffle
import pandas as pd
import math
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=1, help='Random seed.')
parser.add_argument('--epochs', type=int, default=400,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Learning rate.')
parser.add_argument('--weight_decay', type=float, default=1e-7,
                    help='Weight decay.')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

set_seed(args.seed, args.cuda)

nm = 495
nd = 383
nc = 5430

miRNAnumber = np.genfromtxt(r'miRNA number.txt', dtype=str, delimiter='\t')
diseasenumber = np.genfromtxt(r'disease number.txt', dtype=str, delimiter='\t')

DS = np.loadtxt(r'd-d.txt', delimiter=',')
MS = np.loadtxt(r'm-m.txt', delimiter=',')
MS = normalized(MS)
DS = normalized(DS)

DS = torch.from_numpy(DS).float()
MS = torch.from_numpy(MS).float()
A = np.zeros((nm, nd), dtype=float)
ConnectData = np.loadtxt(r'D:\WritingCode\SAEMDA-main\known miRNA-disease associations.txt', dtype=int) - 1

for i in range(nc):
    A[ConnectData[i, 0], ConnectData[i, 1]] = 1

# 取所有未关联样本对的坐标用data0_index保存起来
data0_index = np.argwhere(A == 0)  # (184155,2)
np.savetxt('unknown miRNA-disease association',data0_index,delimiter=' ',fmt='%d')
A = torch.from_numpy(A).float()
# miRNA_feature = torch.cat((A,MS),1)
# disease_feature = torch.cat((A.t(),DS),1)
miRNA_feature = MS
disease_feature = DS

if args.cuda:
    A = A.cuda()
    DS = DS.cuda()
    MS = MS.cuda()
    miRNA_feature = miRNA_feature.cuda()
    disease_feature = disease_feature.cuda()

def train(sgae, y0, adj, epoch):
    optp = torch.optim.Adam(sgae.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    for e in range(1, epoch + 1):
        sgae.train()
        z1, z2, z3, y1, y2, y3 = sgae(adj, y0)
        loss = sgae.my_mse_loss(adj, y0)
        optp.zero_grad()
        loss.backward()
        optp.step()
        sgae.eval()
        with torch.no_grad():
            z1, z2, z3, y1, y2, y3 = sgae(adj,y0)

        if e % 20 == 0 and e != 0:
            print('Epoch %d | Loss: %.4f' % (e, loss.item()))

    return z1, z2, z3


def trainres(mf, df, adjm, adjd):
    # sgaem = SGAE(256, 128, 64, 383+495)
    # sgaed = SGAE(256, 128, 64, 495+383)
    sgaem = SGAE(256, 128, 64, 495 )
    sgaed = SGAE(256, 128, 64, 383 )
    if args.cuda:
        sgaem = sgaem.cuda()
        sgaed = sgaed.cuda()
    zm1, zm2, zm = train(sgaem, mf, adjm, args.epochs)
    zd1, zd2, zd = train(sgaed, df, adjd,args.epochs)
    return zm, zd


zm, zd = trainres(miRNA_feature,disease_feature,MS,DS)
zm = zm.cpu().detach().numpy()
zd = zd.cpu().detach().numpy()
np.savetxt('miRNA embedding',zm,delimiter=',')
np.savetxt('disease embedding',zd,delimiter=',')
