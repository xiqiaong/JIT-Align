import random
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import pickle
from sklearn.model_selection import train_test_split
import pandas as pd
import re
from sklearn.metrics import confusion_matrix, roc_auc_score, matthews_corrcoef, precision_recall_fscore_support, \
    classification_report, auc
import math
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
import logging
import os
import os
print(os.getcwd())
import re
import csv
from openpyxl import Workbook, load_workbook
from .file_sort import files_sort, files_semantic_sort, hunks_semantic_sort, msg_semantic_sort, calculate_vector_similarity

logger = logging.getLogger("my_util")
file_handler = logging.FileHandler("my_util.log", mode="w")
formatter = logging.Formatter("%(asctime)s %(name)s:%(levelname)s:%(message)s")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.setLevel(logging.DEBUG)
logger.propagate = False  

def deduplicate(lst):
    unique_dict = {}  
    for item in lst:
        unique_dict[item] = None  
    return list(unique_dict.keys())  

def clean_commit_message(message):
    """
    去除提交消息中以 'change-id' 开头的部分，只保留前面的内容。
    """
    cleaned_message = re.split(r"change[-\s]*id", message, flags=re.IGNORECASE)[0].strip()
    return cleaned_message

def append_to_csv(file_name, data, header):
    """将数据追加到 CSV 文件中，若文件不存在则创建并写入表头"""
    file_exists = os.path.exists(file_name)
    
    with open(file_name, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        
        if not file_exists and header:
            writer.writerow(header)
        
        writer.writerow(data)

def preprocess_code_line(code, remove_python_common_tokens=False):
    code = code.replace('(', ' ').replace(')', ' ').replace('{', ' ').replace('}', ' ').replace('[', ' ').replace(']',
                                                                                                                  ' ').replace(
        '.', ' ').replace(':', ' ').replace(';', ' ').replace(',', ' ').replace(' _ ', '_')

    code = re.sub('``.*``', '<STR>', code)
    code = re.sub("'.*'", '<STR>', code)
    code = re.sub('".*"', '<STR>', code)
    code = re.sub('\d+', '<NUM>', code)

    code = code.split()
    code = ' '.join(code)
    if remove_python_common_tokens:
        new_code = ''
        python_common_tokens = []
        for tok in code.split():
            if tok not in [python_common_tokens]:
                new_code = new_code + tok + ' '

        return new_code.strip()

    else:
        return code.strip()


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, commit_id, input_ids, input_mask, input_tokens, label):
        self.commit_id = commit_id
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_tokens = input_tokens
        self.label = label


def convert_examples_to_features(item, cls_token='[CLS]', sep_token='[SEP]', sequence_a_segment_id=0,
                                 sequence_b_segment_id=1,
                                 cls_token_segment_id=1, pad_token_segment_id=0, pad_token=0,
                                 mask_padding_with_zero=True, no_abstraction=True):
    commit_id, files, msg, label, tokenizer, args = item
    # print(files)
    label = int(label)
    added_tokens = []
    removed_tokens = []
    # msg_tokens
    if args.code_sequence_mode not in ['random', 'file_origin']:
        msg = clean_commit_message(msg)
    msg_tokens = tokenizer.tokenize(msg)
    # msg_tokens_origin_length = len(msg_tokens)
    msg_tokens = msg_tokens[:min(args.max_msg_length, len(msg_tokens))]
    msg_labels = [0] * len(msg_tokens)  
    # msg_tokens_after_length = len(msg_tokens)
    
    added_labels = []
    removed_labels = []

    # (1) 该提交的所有文件的所有代码行乱序
    if args.code_sequence_mode == 'random':
        file_codes = {'added_code':set(), 'removed_code':set()}
        # pattern = re.compile(r'added_code:(.*?)removed_code:(.*)', re.DOTALL)
        # for f in files:
        #     # print(f+'\n')
        #     matches = pattern.search(f)
        #     if matches:
        #         # print("added_code：", matches.group(1).strip())
        #         # print("removed_code：", matches.group(2).strip())
        #         file_codes['added_code'].add(matches.group(1).strip())
        #         file_codes['removed_code'].add(matches.group(2).strip())
        #     else:
        #         # print("Pattern not found")
        #         logger.info("Pattern not found")
        # logger.info('file_codes')
        # logger.info(file_codes)
        
        # file_codes = files
        
        for filename, changes in files.items():  
            file_label = 1 if changes.get("file_label", 0) == 1 else 0
            
            added_lines = [(code, file_label) for line, code in changes["added_code"].items()]
            removed_lines = [(code, file_label) for line, code in changes["removed_code"].items()]

            file_codes["added_code"].update(added_lines)
            file_codes["removed_code"].update(removed_lines)
    
    # (2) 按照该提交的原文件顺序和原代码顺序
    elif args.code_sequence_mode == 'file_origin':
        file_codes = {'added_code':[], 'removed_code':[]}
        for filename, changes in files.items():  
            file_label = 1 if changes.get("file_label", 0) == 1 else 0
            
            added_lines = [(code, file_label) for line, code in changes["added_code"].items()]
            removed_lines = [(code, file_label) for line, code in changes["removed_code"].items()]

            
            file_codes["added_code"].extend(added_lines)
            file_codes["removed_code"].extend(removed_lines)
    
    # (3) 经过文件扩展名优先级排序
    elif args.code_sequence_mode == 'file_extension_sort':
        files = files_sort(files)
        file_codes = {'added_code':[], 'removed_code':[]}
        for filename, changes in files.items():  
            
            added_lines = [code for line, code in changes["added_code"].items()]
            removed_lines = [code for line, code in changes["removed_code"].items()]

            
            file_codes["added_code"].extend(added_lines)
            file_codes["removed_code"].extend(removed_lines)

    # (4) 与代码提交信息进行匹配排序
    elif args.code_sequence_mode == 'file_semantic_sort':
        files = files_semantic_sort(msg, files, args.code_sequence_settings)
        file_codes = {'added_code':[], 'removed_code':[]}
        for filename, changes in files.items(): 
           
            added_lines = [code for line, code in changes["added_code"].items()]
            removed_lines = [code for line, code in changes["removed_code"].items()]
            
            
            added_lines = deduplicate(added_lines)
            removed_lines = deduplicate(removed_lines)

            
            file_codes["added_code"].extend(added_lines)
            file_codes["removed_code"].extend(removed_lines)

    # (5) 在(3)基础上再进行代码提交信息匹配
    # (6) 以hunk为粒度进行提交信息与提交信息匹配
    elif args.code_sequence_mode == 'hunk_semantic_sort':
        sorted_files = hunks_semantic_sort(msg, files, args.code_sequence_settings)
        file_codes = {'added_code': [], 'removed_code': []}
    
        for filename, sorted_hunks, file_score in sorted_files:
            for hunk_dict, hunk_score in sorted_hunks:  
                
                added_lines = [code for line, code in hunk_dict["added_code"].items()]
                removed_lines = [code for line, code in hunk_dict["removed_code"].items()]
    
                
                file_codes["added_code"].extend(added_lines)
                file_codes["removed_code"].extend(removed_lines)
        
        file_codes["added_code"] = deduplicate(file_codes["added_code"])
        file_codes["removed_code"] = deduplicate(file_codes["removed_code"])

    # (7) 将每个文件的代码变更生成提交消息后再排序
    elif args.code_sequence_mode == 'msg_generation_sort':
        files = msg_semantic_sort(msg, files, args.code_sequence_settings)
        file_codes = {'added_code':[], 'removed_code':[]}
        for filename, changes in files.items():  
            
            added_lines = [code for line, code in changes["added_code"].items()]
            removed_lines = [code for line, code in changes["removed_code"].items()]
            
            # added_lines = deduplicate(added_lines)
            # removed_lines = deduplicate(removed_lines)

            file_codes["added_code"].extend(added_lines)
            file_codes["removed_code"].extend(removed_lines)

    # (8) 向量相似度
    elif args.code_sequence_mode == 'vector_sort':
        files = calculate_vector_similarity(commit_id, files, 'openstack',sort=args.code_sequence_settings)
        file_codes = {'added_code':[], 'removed_code':[]}
        for filename, changes in files.items():  
            file_label = 1 if changes.get("file_label", 0) == 1 else 0
            
            added_lines = [(code, file_label) for line, code in changes["added_code"].items()]
            removed_lines = [(code, file_label) for line, code in changes["removed_code"].items()]

            
            file_codes["added_code"].extend(added_lines)
            file_codes["removed_code"].extend(removed_lines)
       
        file_codes["added_code"] = deduplicate(file_codes["added_code"])
        file_codes["removed_code"] = deduplicate(file_codes["removed_code"])

    if no_abstraction:
        added_codes = [(' '.join(line.split()), file_label) for line, file_label in file_codes['added_code']]
    else:
        added_codes = [(preprocess_code_line(line, False), file_label) for line, file_label in file_codes['added_code']]
    codes = '[ADD]'.join([line for line, _ in added_codes if len(line)])
    
    # logger.info('token前[ADD]:{}'.format(codes))
    
    added_tokens.extend(tokenizer.tokenize(codes))

    for i, (line, file_label) in enumerate(added_codes):
        tokens = tokenizer.tokenize(line)
        added_labels.extend([file_label] * len(tokens))  
        
        if i < len(added_codes) - 1:  
            added_labels.append(0) 

    if no_abstraction:
        removed_codes = [(' '.join(line.split()), file_label) for line, file_label in file_codes['removed_code']]
    else:
        removed_codes = [(preprocess_code_line(line, False), file_label) for line, file_label in file_codes['removed_code']]
    codes = '[DEL]'.join([line for line, _ in removed_codes if len(line)])
    
    # logger.info('token前[DEL]:{}'.format(codes))
    
    removed_tokens.extend(tokenizer.tokenize(codes))

    
    for i, (line, file_label) in enumerate(removed_codes):
        tokens = tokenizer.tokenize(line)
        removed_labels.extend([file_label] * len(tokens))  
        
        if i < len(removed_codes) - 1:  
            removed_labels.append(0) 

    # [MSG]token
    input_tokens = msg_tokens + ['[ADD]'] + added_tokens + ['[DEL]'] + removed_tokens
    input_labels = msg_labels + [0] + added_labels + [0] + removed_labels
    
    # logger.info('input_tokens_before length:{}'.format(len(input_tokens)))
    
    
    input_tokens = input_tokens[:512 - 2]
    # input_labels = input_labels[:512 - 2]

    
    if label == 1:
        if sum(input_labels) != 0:
            truncated_defect_token_count = sum(input_labels[510:])
            truncated_defect_ratio = truncated_defect_token_count / sum(input_labels)

            truncated_data = [commit_id, truncated_defect_token_count, truncated_defect_ratio]
            header = ['commit_id', 'truncated_defect_token_count', 'truncated_defect_ratio']
            append_to_csv("truncated_count.csv", truncated_data, header)
        else:
            logger.warning('check commit {} fix_location vs files_diff!'.format(commit_id))
    # logger.info('input_tokens_after:{}, length: {}'.format(input_tokens, len(input_tokens)))

    

    input_tokens = [tokenizer.cls_token] + input_tokens + [tokenizer.sep_token]
    input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
    input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

    # Zero-pad up to the sequence length.
    padding_length = 512 - len(input_ids)

    # logger.info('padding length: {}'.format(padding_length))

    # cutoff_data = [commit_id, input_tokens[msg_tokens_after_length+1: ], len(input_tokens), msg_tokens_origin_length, len(msg_tokens), len(added_tokens), len(removed_tokens), padding_length] 
    # header = ['commit_id', 'input_tokens_after_msg', 'input_tokens_len', 'msg_length_origin', 'msg_token_length', 'added_token_length', 'removed_token_length', 'padding_length'] 
    
    
    # append_to_csv("cutoff.csv", cutoff_data, header)
    
    input_ids = input_ids + ([pad_token] * padding_length)
    input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
    assert len(input_ids) == 512
    assert len(input_mask) == 512

    return InputFeatures(commit_id=commit_id,
                         input_ids=input_ids,
                         input_mask=input_mask,
                         input_tokens=input_tokens,
                         label=label)


# manual_features_columns = ['la', 'ld', 'nf', 'ns', 'nd', 'entropy', 'ndev',
#                            'lt', 'nuc', 'age', 'exp', 'rexp', 'sexp', 'fix']


def convert_dtype_dataframe(df, feature_name):
    df['fix'] = df['fix'].apply(lambda x: float(bool(x)))
    df = df.astype({i: 'float32' for i in feature_name})
    return df


class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path=None, mode='train'):
        self.examples = []
        self.args = args
        changes_filename = file_path

        data = []
        # print('changes_filename', changes_filename)
        ddata = pd.read_pickle(changes_filename)

        commit_ids, labels, msgs, codes = ddata
        for commit_id, label, msg, files in zip(commit_ids, labels, msgs, codes):
            data.append((commit_id, files, msg, label, tokenizer, args))
        # only use 20% valid data to keep best model
        # convert example to input features
        if mode == 'train':
            random.seed(args.do_seed)
            
            random.shuffle(data)
            random.seed(args.seed)
        self.examples = [convert_examples_to_features(x, no_abstraction=args.no_abstraction) for x in
                         tqdm(data, total=len(data))]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return (torch.tensor(self.examples[item].input_ids),
                torch.tensor(self.examples[item].input_mask),
                torch.tensor(self.examples[item].label))


def get_recall_at_k_percent_effort(percent_effort, result_df_arg, real_buggy_commits):
    cum_LOC_k_percent = (percent_effort / 100) * result_df_arg.iloc[-1]['cum_LOC']
    buggy_line_k_percent = result_df_arg[result_df_arg['cum_LOC'] <= cum_LOC_k_percent]
    buggy_commit = buggy_line_k_percent[buggy_line_k_percent['label'] == 1]
    recall_k_percent_effort = len(buggy_commit) / float(len(real_buggy_commits))

    return recall_k_percent_effort


def eval_metrics(result_df):
    pred = result_df['defective_commit_pred']
    y_test = result_df['label']

    prec, rec, f1, _ = precision_recall_fscore_support(y_test, pred, average='binary')  # at threshold = 0.5
    tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
    #     rec = tp/(tp+fn)

    FAR = fp / (fp + tn)  # false alarm rate
    dist_heaven = math.sqrt((pow(1 - rec, 2) + pow(0 - FAR, 2)) / 2.0)  # distance to heaven

    AUC = roc_auc_score(y_test, result_df['defective_commit_prob'])

    result_df['defect_density'] = result_df['defective_commit_prob'] / result_df['LOC']  # predicted defect density
    result_df['actual_defect_density'] = result_df['label'] / result_df['LOC']  # defect density

    result_df = result_df.sort_values(by='defect_density', ascending=False)
    actual_result_df = result_df.sort_values(by='actual_defect_density', ascending=False)
    actual_worst_result_df = result_df.sort_values(by='actual_defect_density', ascending=True)

    result_df['cum_LOC'] = result_df['LOC'].cumsum()
    actual_result_df['cum_LOC'] = actual_result_df['LOC'].cumsum()
    actual_worst_result_df['cum_LOC'] = actual_worst_result_df['LOC'].cumsum()

    real_buggy_commits = result_df[result_df['label'] == 1]

    label_list = list(result_df['label'])

    all_rows = len(label_list)

    # find Recall@20%Effort
    cum_LOC_20_percent = 0.2 * result_df.iloc[-1]['cum_LOC']
    buggy_line_20_percent = result_df[result_df['cum_LOC'] <= cum_LOC_20_percent]
    buggy_commit = buggy_line_20_percent[buggy_line_20_percent['label'] == 1]
    recall_20_percent_effort = len(buggy_commit) / float(len(real_buggy_commits))

    # find Effort@20%Recall
    buggy_20_percent = real_buggy_commits.head(math.ceil(0.2 * len(real_buggy_commits)))
    buggy_20_percent_LOC = buggy_20_percent.iloc[-1]['cum_LOC']
    effort_at_20_percent_LOC_recall = int(buggy_20_percent_LOC) / float(result_df.iloc[-1]['cum_LOC'])

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

        predicted_recall_at_percent_effort_list.append(predicted_recall_k_percent_effort)
        actual_recall_at_percent_effort_list.append(actual_recall_k_percent_effort)
        actual_worst_recall_at_percent_effort_list.append(actual_worst_recall_k_percent_effort)

    p_opt = 1 - ((auc(percent_effort_list, actual_recall_at_percent_effort_list) -
                  auc(percent_effort_list, predicted_recall_at_percent_effort_list)) /
                 (auc(percent_effort_list, actual_recall_at_percent_effort_list) -
                  auc(percent_effort_list, actual_worst_recall_at_percent_effort_list)))

    return f1, AUC, recall_20_percent_effort, effort_at_20_percent_LOC_recall, p_opt


def load_change_metrics_df(data_dir, mode='train'):
    change_metrics = pd.read_pickle(data_dir)
    feature_name = ["ns", "nd", "nf", "entropy", "la", "ld", "lt", "fix", "ndev", "age", "nuc", "exp", "rexp", "sexp"]
    change_metrics = convert_dtype_dataframe(change_metrics, feature_name)

    return change_metrics[['commit_hash'] + feature_name]


def eval_result(result_path, features_path):
    RF_result = pd.read_csv(result_path, sep='\t')

    RF_result.columns = ['test_commit', 'defective_commit_prob', 'defective_commit_pred', 'label']  # for new result

    test_commit_metrics = load_change_metrics_df(features_path, 'test')[['commit_hash', 'la', 'ld']]
    RF_df = pd.DataFrame()
    RF_df['commit_id'] = RF_result['test_commit']
    RF_df = pd.merge(RF_df, test_commit_metrics, left_on='commit_id', right_on='commit_hash', how='inner')
    RF_df = RF_df.drop('commit_hash', axis=1)
    RF_df['LOC'] = RF_df['la'] + RF_df['ld']
    RF_result = pd.merge(RF_df, RF_result, how='inner', left_on='commit_id', right_on='test_commit')
    f1, auc, recall_20_percent_effort, effort_at_20_percent_LOC_recall, p_opt = eval_metrics(
        RF_result)
    logging.info(
        'F1: {:.4f}, AUC: {:.4f}, PCI@20%LOC: {:.4f}, Effort@20%Recall: {:.4f}, POpt: {:.4f}'.format(
            f1, auc, recall_20_percent_effort, effort_at_20_percent_LOC_recall, p_opt))


def get_line_level_metrics(line_score, label):
    scaler = MinMaxScaler()
    line_score = scaler.fit_transform(np.array(line_score).reshape(-1, 1))  # cannot pass line_score as list T-T
    pred = np.round(line_score)

    line_df = pd.DataFrame()
    line_df['scr'] = [float(val) for val in list(line_score)]
    line_df['label'] = label
    line_df = line_df.sort_values(by='scr', ascending=False)
    line_df['row'] = np.arange(1, len(line_df) + 1)

    real_buggy_lines = line_df[line_df['label'] == 1]

    top_10_acc = 0
    top_5_acc = 0

    if len(real_buggy_lines) < 1:
        IFA = len(line_df)
        top_20_percent_LOC_recall = 0
        effort_at_20_percent_LOC_recall = math.ceil(0.2 * len(line_df))

    else:
        IFA = line_df[line_df['label'] == 1].iloc[0]['row'] - 1
        label_list = list(line_df['label'])

        all_rows = len(label_list)

        # find top-10 accuracy
        if all_rows < 10:
            top_10_acc = np.sum(label_list[:all_rows]) / len(label_list[:all_rows])
        else:
            top_10_acc = np.sum(label_list[:10]) / len(label_list[:10])

        # find top-5 accuracy
        if all_rows < 5:
            top_5_acc = np.sum(label_list[:all_rows]) / len(label_list[:all_rows])
        else:
            top_5_acc = np.sum(label_list[:5]) / len(label_list[:5])

        # find recall
        LOC_20_percent = line_df.head(int(0.2 * len(line_df)))
        buggy_line_num = LOC_20_percent[LOC_20_percent['label'] == 1]
        top_20_percent_LOC_recall = float(len(buggy_line_num)) / float(len(real_buggy_lines))

        # find effort @20% LOC recall

        buggy_20_percent = real_buggy_lines.head(math.ceil(0.2 * len(real_buggy_lines)))
        buggy_20_percent_row_num = buggy_20_percent.iloc[-1]['row']
        effort_at_20_percent_LOC_recall = int(buggy_20_percent_row_num) / float(len(line_df))

    return IFA, top_20_percent_LOC_recall, effort_at_20_percent_LOC_recall, top_10_acc, top_5_acc


def create_path_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)
