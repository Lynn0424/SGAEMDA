# 开发时间  2022/6/7 20:36
import random
from numpy import interp
from sklearn import metrics
import joblib
from sklearn.metrics import auc
from sklearn.metrics import precision_score, recall_score, f1_score,accuracy_score
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
# from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from utils import *

random.seed(10)

ConnectData = np.loadtxt(r'known miRNA-disease associations.txt', dtype=int) - 1
UnknownData = np.loadtxt(r'unknown miRNA-disease association',dtype=int)
miRNAnumber = np.genfromtxt(r'miRNA number.txt', dtype=str, delimiter='\t')
diseasenumber = np.genfromtxt(r'disease number.txt', dtype=str, delimiter='\t')

M_fea = np.loadtxt(r'D:\WritingCode\Try\miRNA embedding', delimiter=',')
D_fea = np.loadtxt(r'D:\WritingCode\Try\disease embedding', delimiter=',')
DS = np.loadtxt(r'd-d.txt', delimiter=',')
MS = np.loadtxt(r'm-m.txt', delimiter=',')



A = np.zeros((495, 383), dtype=float)
for i in range(5430):
    A[ConnectData[i, 0], ConnectData[i, 1]] = 1
# data0_index = np.argwhere(A == 0)
M_fea = np.hstack((M_fea,A))
D_fea = np.hstack((D_fea,A.transpose()))


# 加载正样本
# Zheng = np.zeros((5430, 128), dtype=float)
Zheng = np.zeros((5430, 1006), dtype=float)
for i in range(5430):
    zheng = []
    zheng.extend(M_fea[ConnectData[i, 0]])
    zheng.extend(D_fea[ConnectData[i, 1]])
    Zheng[i] = zheng

# # 随机采样负样本
suiji = random.sample(list(UnknownData), 5430)

# Fu = np.zeros((5430, 128), dtype=float)  # 负样本 维度为(5340,128)
Fu = np.zeros((5430, 1006), dtype=float)
for i in range(5430):
    fu = []
    fu.extend(M_fea[suiji[i][0]])
    fu.extend(D_fea[suiji[i][1]])
    Fu[i] = fu

# 创建标签 labels 维度为(10860,2)，前5430为[1],后5430为[0]
labels = []
for i in range(5430):
    labels.append([1])
for i in range(5430):
    labels.append([0])
labels = np.array(labels, dtype=int)
labels = labels.flatten()

# 创建特征，将正负样本合到一起 feature维度为(10860,256)
feature = np.vstack((Zheng, Fu))
feature = np.array(feature, dtype=float)


def train(X, Y):
    X, Y = shuffle(X, Y, random_state=10)
    '''MLP'''
    clf = MLPClassifier((128,64),solver='adam',alpha=1e-5,random_state=1,max_iter=1000)
    kf = KFold(n_splits=5)
    print("开始训练!")
    tprs = []
    AUC_list = []
    mean_fpr = np.linspace(0,1,100)
    acc_list = []
    p_list = []
    r_list = []
    f1_list = []
    AUPR_list = []
    i = 1
    for train_index, test_index in kf.split(X, Y):
        X_train = X[train_index]
        Y_train = Y[train_index]
        X_test = X[test_index]
        Y_test = Y[test_index]
        clf.fit(X_train, Y_train)
        predict_value = clf.predict_proba(X_test)[:, 1] # 输出[ 负的概率], 正的概率] 格式的数据,取第二列 正的概率
        # print(clf.predict_proba(X_test).shape)
        # acc 的label必须是0或1
        acc_pre = np.argmax(clf.predict_proba(X_test),axis=1)
        # AUC = metrics.roc_auc_score(Y_test, predict_value)
        '''画图'''
        fpr, tpr, _ = roc_curve(Y_test,predict_value,drop_intermediate=False)
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr,tpr)
        plt.plot(fpr,tpr,label='fold {0:d} (AUC = {1:.4f})'.format(i,roc_auc),linestyle='--',alpha=0.4)
        # joblib.dump(clf, './save/model_%d.pkl' % i)
        i = i + 1
        precision, recall, _ = precision_recall_curve(Y_test, predict_value)
        AUC_list.append(roc_auc)
        AUCPR = auc(recall, precision)
        AUPR_list.append(AUCPR)
        p = precision_score(Y_test, predict_value.round())
        p_list.append(p)
        r = recall_score(Y_test, predict_value.round())
        r_list.append(r)
        f1 = f1_score(Y_test, predict_value.round())
        f1_list.append(f1)
        acc = accuracy_score(Y_test, acc_pre)
        acc_list.append(acc)

    auc_l = np.array(AUC_list)
    aupr_l = np.array(AUPR_list)
    p_l = np.array(p_list)
    r_l = np.array(r_list)
    f1_l = np.array(f1_list)
    acc_l = np.array(acc_list)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    # mean_auc = auc(mean_fpr, mean_tpr)
    # std_auc = np.std(AUC_list)

    print('AUROC = %.4f +- %.4f | AUPR = %.4f +- %.4f' % (auc_l.mean(), auc_l.std(), aupr_l.mean(),
                                                        aupr_l.std()))
    print('Precision = %.4f +- %.4f' % (p_l.mean(), p_l.std()))
    print('Recall = %.4f +- %.4f' % (r_l.mean(), r_l.std()))
    print('F1_score = %.4f +- %.4f' % (f1_l.mean(), f1_l.std()))
    print('Accuracy = %.4f +- %.4f' % (acc_l.mean(), acc_l.std()))

    plt.plot(mean_fpr, mean_tpr, label=r'Mean AUC (AUC = {0:.4f} $\pm$ {1:0.4f})'.format(auc_l.mean(),auc_l.std()),
            color='BlueViolet',alpha=0.9)
    plt.plot([0, 1], [0, 1], '--',color='black',alpha=0.4)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend(loc='lower right')
    plt.savefig('./figure/5foldCV_AUC.jpg',dpi=1200,bbox_inche='tight')
    plt.show()
if __name__ == "__main__":
    train(feature, labels)




