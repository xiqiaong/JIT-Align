from itertools import groupby
import csv
from collections import Counter
from sklearn.metrics import average_precision_score, auc, precision_recall_curve, roc_curve, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
import numpy as np
import argparse

np.random.seed(10)

def read_csv_1 (fname):

    label = []
    # la = []
    pred = []
    with open(fname, 'r') as f:
        reader = csv.reader(f)
        i = 0
        for line in reader:
            i += 1
            if i == 1:
                continue
            label.append(int(line[0]))
            # la.append(int(line[1]))
            pred.append(float(line[1]))
            
        
        
    #print(len(pred), len(label), len(la))
    return label, pred


def read_csv_2 (fname):

    label = []
    # la = []
    pred = []
    with open(fname, 'r') as f:
        reader = csv.reader(f)
        i = 0
        for line in reader:
            i += 1
            if i == 1:
                continue
            label.append(line[0])
            pred.append(float(line[1]))
            
              
    #print(len(pred), len(label))
    return label, pred



def eval_(y_true,y_pred, thresh=None):
    

    #print('size:', len(y_true), len(y_pred))
    auc = roc_auc_score(y_true=y_true, y_score=y_pred)
    #auc_pc(y_true, y_pred)
    if thresh != None:
        y_pred = [ 1.0 if p> thresh else 0.0 for p in y_pred]
        
    
    acc = accuracy_score(y_true=y_true, y_pred=y_pred)
    prc = precision_score(y_true=y_true, y_pred=y_pred)
    rc = recall_score(y_true=y_true, y_pred=y_pred)
    f1 = f1_score(y_true=y_true, y_pred=y_pred)
    print('AUC:', auc)
    
    
## AUC-PC
# predict class values
def auc_pc(label, pred):
    lr_probs = np.array(pred)
    testy = np.array([float(l) for l in label])
    no_skill = len(testy[testy==1]) / len(testy)

    lr_precision, lr_recall, _ = precision_recall_curve(testy, lr_probs)

    lr_auc = auc(lr_recall, lr_precision)

    return lr_auc


parser = argparse.ArgumentParser()
parser.add_argument('-project', type=str, default='openstack')
args = parser.parse_args()


data_dir1 = "./semantic/pred_scores/"
data_dir2 = "./artificial/pred_scores/"

project = args.project

#Com
com_ = data_dir1+ 'test_com_' + project + '.csv'

#Sim
sim_ = data_dir2 + 'test_sim_' + project +'.csv'


##LAPredict or Kamei's 14
label, pred= read_csv_2 (sim_)


##codebert 
label_, pred_ = read_csv_2 (com_)


## Simple add
pred2 = [ pred_[i] + pred[i] for i in range(len(pred_))]
#print(len(pred2), len(label_))
auc2 = roc_auc_score(y_true=np.array(label_),  y_score=np.array(pred2))
#print('\n SimCom: ')
mean_pred = float(sum(pred2)/len(pred2))
#eval_(y_true=np.array(label_),  y_pred=np.array(pred2), thresh = mean_pred )
pc_ = auc_pc(label_, pred2)

t = 1
real_label = [float(l) for l in label_]
real_pred = [1 if p > t else 0 for p in pred2]
f1_ = f1_score(y_true=real_label,  y_pred=real_pred)
mcc = matthews_corrcoef(real_label, real_pred)
recall = recall_score(real_label, real_pred, average='binary')
precision = precision_score(real_label, real_pred, average='binary')
print("AUC-ROC:{}  AUC-PR:{}  F1-Score:{}  MCC:{}  Recall:{}  Precision:{}".format(auc2, pc_, f1_, mcc, recall, precision))

