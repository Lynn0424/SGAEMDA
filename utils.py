import numpy as np
import torch
import argparse
from sklearn.preprocessing import minmax_scale,scale
from sklearn.metrics import roc_curve,roc_auc_score,average_precision_score,precision_recall_curve,auc
import matplotlib.pyplot as plt

# 设置随机数种子
def set_seed(seed,cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)

# 拉普拉斯归一化
def normalized(wmat):
    deg = np.diag(np.sum(wmat,axis=0))
    degpow = np.power(deg,-0.5)
    degpow[np.isinf(degpow)] = 0
    W = np.dot(np.dot(degpow,wmat),degpow)
    return W


def get_from_idx(idxs,pairs):
    data=[]
    for i in idxs:
        data.append(pairs[i])
    return data

def make_data(pairs):
    datax = []
    label = []
    for pair in pairs:
        x=[]
        x.append(pair[0])
        x.append(pair[1])
        label.append(pair[2])
        datax.append(x)
    return torch.LongTensor(datax), torch.LongTensor(label)

def show_auc(ymat):
# def show_auc(ymat,data):
    # path = 'Dataset'+str(data)
    ldi = np.loadtxt('m-d.txt',delimiter=',')
    y_true = ldi.flatten()
    ymat = ymat.flatten()
    fpr,tpr,rocth = roc_curve(y_true,ymat)
    auroc = auc(fpr,tpr)

    np.savetxt('roc.txt',np.vstack((fpr,tpr)),fmt='%10.5f',delimiter=',')

    precision,recall,prth = precision_recall_curve(y_true,ymat)
    aupr = auc(recall,precision)

    np.savetxt('pr.txt',np.vstack((recall,precision)),fmt='%10.5f',delimiter=',')

    print('AUROC= %.4f | AUPR= %.4f' % (auroc,aupr))


    rocdata = np.loadtxt('roc.txt',delimiter=',')
    prdata = np.loadtxt('pr.txt',delimiter=',')
    # plt.figure()
    # plt.plot(rocdata[0],rocdata[1])
    # plt.plot(prdata[0],prdata[1])
    # plt.show()
    return auroc,aupr