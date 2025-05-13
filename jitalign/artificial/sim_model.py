import pickle
import math
import random
import time
import argparse

from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score, roc_curve, auc, roc_auc_score, average_precision_score,  precision_recall_curve, matthews_corrcoef
import pandas as pd
import numpy as np
import torch.nn as nn
import torch

from collections import Counter
from itertools import groupby

from LR import LR
from DBN import DBN

#import xgboost as xgb
from imblearn.over_sampling import SMOTE 
from imblearn.under_sampling import RandomUnderSampler 
from imblearn.over_sampling import RandomOverSampler
from xgboost import XGBClassifier
import logging
from datetime import datetime
logging.basicConfig(filename=f"../Log/sim{datetime.now().strftime('%Y%m%d%H')}.log", level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def auc_pc(label, pred):
    lr_probs = np.array(pred)
    testy = np.array([float(l) for l in label])

    lr_precision, lr_recall, _ = precision_recall_curve(testy, lr_probs)
    lr_auc = auc(lr_recall, lr_precision)
    #print('AUC-PR:  auc=%.3f' % ( lr_auc))
    return lr_auc


def train_and_evl(data, label, args):
    size = int(label.shape[0]*0.2)
    auc_ = []

    for i in range(5):
        idx = size * i
        X_e = data[idx:idx+size]
        y_e = label[idx:idx+size]

        X_t = np.vstack((data[:idx], data[idx+size:]))
        y_t = np.hstack((label[:idx], label[idx+size:]))


        model = LogisticRegression(max_iter=7000).fit(X_t, y_t)
        y_pred = model.predict_proba(X_e)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_true=y_e, y_score=y_pred, pos_label=1)
        auc_.append(auc(fpr, tpr))

    return np.mean(auc_)


def replace_value_dataframe(df):
    df = df.replace({True: 1, False: 0})
    # df = df.fillna(df.mean()) ERROR
    if df.isnull().values.any():
        logging.info("Warning:there is nan in df")
        logging.info("position is {}".format(df.isnull().sum()))
        df = df.fillna(0)
    #print(df.keys())
    if args.drop:
        df = df.drop(columns=[args.drop])
    elif args.only:
        df = df[['_id','date','bug','__'] + args.only]
        #print('new index:', df.keys())
        logging.info('dataframe keys:{}'.format(df.keys()))
    return df.values


def get_features(data):
    # return the features of yasu data
    logging.info("get_feature for example:{}".format(data[0, 4:]))
    logging.info("get_feature for example:{}".format(data[1, 4:]))
    return data[:, 4:]


def get_ids(data):
    # return the labels of yasu data
    # flatten()
    logging.info("get_ids for example:{}".format(data[0, 0:1]))
    logging.info("get_ids for example:{}".format(data[1, 0:1]))
    return data[:, 0:1].flatten().tolist()


def get_label(data):
    logging.info("get_label for example:{}".format(data[0, 2:3]))
    logging.info("get_label for example:{}".format(data[1, 2:3]))
    data = data[:, 2:3].flatten().tolist()
    data = [1 if int(d) > 0 else 0 for d in data]
    return data


def load_df_yasu_data(path_data, flag=None):

    data = pd.read_csv(path_data)
    data = replace_value_dataframe(df=data)
    ids, labels, features = get_ids(data=data), get_label(data=data), get_features(data=data)
    indexes = list()
    cnt_noexits = 0

    for i in range(0, len(ids)):

        try:
            indexes.append(i)
        except FileNotFoundError:
            #print('File commit id no exits', ids[i], cnt_noexits)
            cnt_noexits += 1


    ids = [ids[i] for i in indexes]
    labels = [labels[i] for i in indexes]
    features = features[indexes]
    return (ids, np.array(labels), features)


def load_yasu_data(args):
    train_path_data = '/root/jitalign/Data_Extraction/git_base/datasets/{}/k_feature/{}_train.csv'.format(args.project, args.data)
    test_path_data = '/root/jitalign/Data_Extraction/git_base/datasets/{}/k_feature/{}_test.csv'.format(args.project, args.data)
    train, test = load_df_yasu_data(train_path_data, 'train'), load_df_yasu_data(test_path_data, 'test')
    return train, test

def evaluation_metrics(y_true, y_pred):
    fpr, tpr, thresholds = roc_curve(y_true=y_true, y_score=y_pred, pos_label=1)
    auc_ = auc(fpr, tpr)
    
    y_pred = [1 if p >= 0.5 else 0 for p in y_pred]
    acc = accuracy_score(y_true=y_true, y_pred=y_pred)
    prc = precision_score(y_true=y_true, y_pred=y_pred)
    rc = recall_score(y_true=y_true, y_pred=y_pred)
    f1 = 2 * prc * rc / (prc + rc)

    mcc = matthews_corrcoef(y_true, y_pred)
    
    return acc, prc, rc, f1, auc_, mcc



def balance_pos_neg_in_training(X_train,y_train):

    #print(sorted(Counter(y_train).items()))
    
    #ros = RandomOverSampler(random_state=42)
    
    sm = SMOTE(random_state=42)
    # rus = RandomUnderSampler(random_state=42)
    print('y_train',type(y_train))
    #X_resampled, y_resampled = ros.fit_resample(X_train, y_train)
    X_resampled, y_resampled = sm.fit_resample(X_train, y_train)
    # X_resampled, y_resampled = rus.fit_resample(X_train, y_train)
    
    #print(sorted(Counter(y_resampled).items()))
    return X_resampled, y_resampled

def mini_batches_update(X, Y, mini_batch_size=64, seed=0):
    m = X.shape[0]  # number of training examples
    mini_batches = list()
    np.random.seed(seed)

    # Step 1: No shuffle (X, Y)
    shuffled_X, shuffled_Y = X, Y
    Y = Y.tolist()
    Y_pos = [i for i in range(len(Y)) if Y[i] == 1]
    Y_neg = [i for i in range(len(Y)) if Y[i] == 0]

    # Step 2: Randomly pick mini_batch_size / 2 from each of positive and negative labels
    num_complete_minibatches = int(math.floor(m / float(mini_batch_size))) + 1
    for k in range(0, num_complete_minibatches):
        indexes = sorted(
            random.sample(Y_pos, int(mini_batch_size / 2)) + random.sample(Y_neg, int(mini_batch_size / 2)))
        mini_batch_X, mini_batch_Y = shuffled_X[indexes], shuffled_Y[indexes]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    return mini_batches

def DBN_JIT(train_features, train_labels, test_features, test_labels, hidden_units=[20, 12, 12], num_epochs_LR=200):
    # training DBN model
    #################################################################################################
    starttime = time.time()
    dbn_model = DBN(visible_units=train_features.shape[1],
                    hidden_units=hidden_units,
                    use_gpu=False)
    dbn_model.train_static(train_features, train_labels, num_epochs=10)
    # Finishing the training DBN model
    # print('---------------------Finishing the training DBN model---------------------')
    # using DBN model to construct features
    DBN_train_features, _ = dbn_model.forward(train_features)
    DBN_test_features, _ = dbn_model.forward(test_features)
    DBN_train_features = DBN_train_features.numpy()
    DBN_test_features = DBN_test_features.numpy()

    train_features = np.hstack((train_features, DBN_train_features))
    test_features = np.hstack((test_features, DBN_test_features))


    if len(train_labels.shape) == 1:
        num_classes = 1
    else:
        num_classes = train_labels.shape[1]
    # lr_model = LR(input_size=hidden_units, num_classes=num_classes)
    lr_model = LR(input_size=train_features.shape[1], num_classes=num_classes)
    optimizer = torch.optim.Adam(lr_model.parameters(), lr=0.00001)
    steps = 0
    batches_test = mini_batches(X=test_features, Y=test_labels)
    for epoch in range(1, num_epochs_LR + 1):
        # building batches for training model
        batches_train = mini_batches_update(X=train_features, Y=train_labels)
        for batch in batches_train:
            x_batch, y_batch = batch
            x_batch, y_batch = torch.tensor(x_batch).float(), torch.tensor(y_batch).float()

            optimizer.zero_grad()
            predict = lr_model.forward(x_batch)
            loss = nn.BCELoss()
            loss = loss(predict, y_batch)
            loss.backward()
            optimizer.step()

            # steps += 1
            # if steps % 100 == 0:
            #     print('\rEpoch: {} step: {} - loss: {:.6f}'.format(epoch, steps, loss.item()))

    endtime = time.time()
    dtime = endtime - starttime
    # print("Train Time: %.8s s" % dtime)  #显示到微秒 
    logging.info("Train Time: {} s".format(dtime))

    starttime = time.time()
    y_pred, lables = lr_model.predict(data=batches_test)
    endtime = time.time()
    dtime = endtime - starttime
    # print("Eval Time: %.8s s" % dtime)  #显示到微秒 
    logging.info("Eval Time: {} s".format(dtime))
    return y_pred

def mini_batches(X, Y, mini_batch_size=64, seed=0):
    m = X.shape[0]  # number of training examples
    mini_batches = list()
    np.random.seed(seed)

    # Step 1: No shuffle (X, Y)
    shuffled_X, shuffled_Y = X, Y

    # Step 2: Partition (X, Y). Minus the end case.
    # number of mini batches of size mini_batch_size in your partitioning
    num_complete_minibatches = int(math.floor(m / float(mini_batch_size)))

    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        if len(Y.shape) == 1:
            mini_batch_Y = shuffled_Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size]
        else:
            mini_batch_Y = shuffled_Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size: m, :]
        if len(Y.shape) == 1:
            mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size: m]
        else:
            mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size: m, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    return mini_batches

def baseline_algorithm(train, test, algorithm, only=False):
    _, y_train, X_train = train
    _, y_test, X_test = test

    ##over/under sample
    X_train,y_train = balance_pos_neg_in_training(X_train,y_train)
    # print(X_train[0,])
    #X_train, X_test = preprocessing.scale(X_train), preprocessing.scale(X_test)    
    #X_train1, X_test1 = scaler.transform(X_train),scaler.transform(X_test)
    #X_train, X_test = preprocessing.scale(X_train), preprocessing.scale(X_test)
    #assert(X_train.all()==X_train1.all())
    acc, prc, rc, f1, auc_ = 0, 0, 0, 0, 0
    if algorithm == 'rf':

        model = RandomForestClassifier(n_estimators=100, random_state=5).fit(X_train, y_train) 
        
        # predict_proba能返回每一类别的预测值，y_pred取True的概率
        y_pred = model.predict_proba(X_test)[:, -1]    

        acc, prc, rc, f1, auc_, mcc = evaluation_metrics(y_true=y_test, y_pred=y_pred)

        # 只有指定only并且不是跨项目时，才做五折交叉验证，为什么只修改AUC的结果
        if only and not "cross" in args.data:
            #print('Here ********** 5 fold cross valid')
            logging.info("Here 5 fold cross valid")
            auc_ = train_and_evl(X_train, y_train, args)
        #print('Accuracy: %f -- Precision: %f -- Recall: %f -- F1: %f -- AUC: %f' % (acc, prc, rc, f1, auc_))
        logging.info('Accuracy: {} -- Precision: {} -- Recall: {}-- F1: {}-- AUC: {}-- MCC: {}'.format(acc, prc, rc, f1, auc_, mcc))
    elif algorithm =='dbn':
        y_pred = DBN_JIT(X_train, y_train, X_test, y_test)
        acc, prc, rc, f1 = 0, 0, 0, 0
        acc, prc, rc, f1, auc_ = evaluation_metrics(y_true=y_test, y_pred=y_pred)
        #print('Accuracy: %f -- Precision: %f -- Recall: %f -- F1: %f -- AUC: %f' % (acc, prc, rc, f1, auc_))
        logging.info('Accuracy: {} -- Precision: {} -- Recall: {}-- F1: {}-- AUC: {}',format(acc, prc, rc, f1, auc_))
    elif algorithm == 'lr':
        model = LogisticRegression(max_iter=7000).fit(X_train, y_train)

        y_pred = model.predict_proba(X_test)[:, -1]

        acc, prc, rc, f1, auc_, mcc = evaluation_metrics(y_true=y_test, y_pred=y_pred)
        if only and not "cross" in args.data:
            auc_ = train_and_evl(X_train, y_train, args)
        # print('Accuracy: %f -- Precision: %f -- Recall: %f -- F1: %f -- AUC: %f' % (acc, prc, rc, f1, auc_))
        logging.info('Accuracy: {} -- Precision: {} -- Recall: {}-- F1: {}-- AUC: {}--MCC: {}'.format(acc, prc, rc, f1, auc_, mcc))
    else:
        print('You need to give the correct algorithm name')
        return

    return y_test, y_pred 

# from CCT5
def convert_dtype_dataframe(df, feature_name):
    df = df.astype({i: 'float32' for i in feature_name})
    return df

def get_recall_at_k_percent_effort(percent_effort, result_df_arg, real_buggy_commits):
    cum_LOC_k_percent = (percent_effort / 100) * \
        result_df_arg.iloc[-1]['cum_LOC']
    buggy_line_k_percent = result_df_arg[result_df_arg['cum_LOC']
                                         <= cum_LOC_k_percent]
    buggy_commit = buggy_line_k_percent[buggy_line_k_percent['label'] == 1]
    recall_k_percent_effort = len(
        buggy_commit) / float(len(real_buggy_commits))

    return recall_k_percent_effort

def eval_R20E_E20R_Popt(test_features, gold, prob):
    test_features = test_features[['_id', 'la', 'ld']]
    print(len(gold))
    print(len(prob))
    test_features['label'] = gold
    test_features = convert_dtype_dataframe(test_features, ['la', 'ld'])
    test_features['LOC'] = test_features['la'] + test_features['ld']

    loc_sum = sum(test_features['LOC'])
    test_features['defective_commit_prob'] = prob
    test_features['defect_density'] = test_features['defective_commit_prob'] / \
        test_features['LOC']  # predicted defect density
    test_features['actual_defect_density'] = test_features['label'] / \
        test_features['LOC']  # defect density

    result_df = test_features.sort_values(by='defect_density', ascending=False)
    actual_result_df = result_df.sort_values(
        by='actual_defect_density', ascending=False)
    actual_worst_result_df = result_df.sort_values(
        by='actual_defect_density', ascending=True)
    result_df['cum_LOC'] = result_df['LOC'].cumsum()
    actual_result_df['cum_LOC'] = actual_result_df['LOC'].cumsum()
    actual_worst_result_df['cum_LOC'] = actual_worst_result_df['LOC'].cumsum()
    real_buggy_commits = result_df[result_df['label'] == 1]

    # find Recall@20%Effort
    cum_LOC_20_percent = 0.2 * loc_sum
    buggy_line_20_percent = result_df[result_df['cum_LOC']
                                      <= cum_LOC_20_percent]
    buggy_commit = buggy_line_20_percent[buggy_line_20_percent['label'] == 1]
    recall_20_percent_effort = len(
        buggy_commit) / float(len(real_buggy_commits))

    # find Effort@20%Recall
    buggy_20_percent = real_buggy_commits.head(
        math.ceil(0.2 * len(real_buggy_commits)))
    buggy_20_percent_LOC = buggy_20_percent.iloc[-1]['cum_LOC']
    effort_at_20_percent_LOC_recall = int(
        buggy_20_percent_LOC) / float(result_df.iloc[-1]['cum_LOC'])

    # find P_opt
    percent_effort_list = []
    predicted_recall_at_percent_effort_list = []
    actual_recall_at_percent_effort_list = []
    actual_worst_recall_at_percent_effort_list = []

    for percent_effort in np.arange(10, 101, 10):
        predicted_recall_k_percent_effort = get_recall_at_k_percent_effort(percent_effort, result_df,
                                                                           real_buggy_commits)
        actual_recall_k_percent_effort = get_recall_at_k_percent_effort(percent_effort, actual_result_df,
                                                                        real_buggy_commits)
        actual_worst_recall_k_percent_effort = get_recall_at_k_percent_effort(percent_effort, actual_worst_result_df,
                                                                              real_buggy_commits)

        percent_effort_list.append(percent_effort / 100)

        predicted_recall_at_percent_effort_list.append(
            predicted_recall_k_percent_effort)
        actual_recall_at_percent_effort_list.append(
            actual_recall_k_percent_effort)
        actual_worst_recall_at_percent_effort_list.append(
            actual_worst_recall_k_percent_effort)

    p_opt = 1 - ((auc(percent_effort_list, actual_recall_at_percent_effort_list) -
                  auc(percent_effort_list, predicted_recall_at_percent_effort_list)) /
                 (auc(percent_effort_list, actual_recall_at_percent_effort_list) -
                     auc(percent_effort_list, actual_worst_recall_at_percent_effort_list)))

    return recall_20_percent_effort, effort_at_20_percent_LOC_recall, p_opt

parser = argparse.ArgumentParser()

parser.add_argument('-project', type=str,
                    default='openstack')
parser.add_argument('-data', type=str,
                    default='k')
parser.add_argument('-algorithm', type=str,
                    default='lr')
parser.add_argument('-drop', type=str,
                    default='')
parser.add_argument('-only', nargs='+',
                    default=[])


args = parser.parse_args()

# only = False

train, test = load_yasu_data(args)
labels, predicts = baseline_algorithm(train=train, test=test, algorithm=args.algorithm, only=args.only)
auc_pc_score = auc_pc(labels, predicts)
auc_roc = roc_auc_score(y_true=labels,  y_score=predicts)

df = pd.DataFrame({'label': labels, 'pred': predicts})
df.to_csv('./pred_scores/test_sim_' + args.project + '.csv', index=False, sep=',')
    















