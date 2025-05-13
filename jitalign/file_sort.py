import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import re
from Levenshtein import distance
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
from gensim import corpora
from gensim.models import LdaModel
import math
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import string
import hashlib
import json
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config, T5Model
import torch
from tqdm import tqdm
import faiss
import numpy as np
from scipy.spatial.distance import cosine, mahalanobis
import csv
from sklearn.cluster import KMeans

stop_words = set(stopwords.words('english'))

def split_camel_snake(text):
    """
    处理驼峰命名法和蛇形命名法，将变量拆分为单词序列
    """
    
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    
    
    text = text.replace("_", " ")
    
    return text

def preprocess_text(text):
   
    text = split_camel_snake(text)

    
    words = word_tokenize(text)

    
    processed_words = [
        word.lower() for word in words
        if word.lower() not in stop_words and word not in string.punctuation
    ]
    
    return " ".join(processed_words)


def clean_commit_message(message):
    """
    去除提交消息中以 'change-id' 开头的部分，只保留前面的内容。
    """
    
    cleaned_message = re.split(r"change[-\s]*id", message, flags=re.IGNORECASE)[0].strip()
    return cleaned_message

def calculate_similarity_cosine(text1, text2):
    """
    计算两个文本之间的语义相似度（余弦相似度）。
    """
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return similarity

def calculate_similarity_jaccard(text1, text2):
    """
    计算两个文本之间的 Jaccard 相似度。
    """
    
    set1 = set(text1.split())
    set2 = set(text2.split())
    
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    
    return intersection / union if union != 0 else 0

def calculate_similarity_levenshtein(text1, text2):
    """
    使用 Levenshtein 编辑距离计算文本相似度。
    """
    edit_dist = distance(text1, text2)  # 计算编辑距离
    max_len = max(len(text1), len(text2))
    return 1 - edit_dist / max_len if max_len > 0 else 0  # 转为相似度

def calculate_similarity_bm25(msg, code_content):
    """
    使用 BM25 计算代码与提交消息的相似度。
    """
   
    msg_tokens = word_tokenize(msg)
    code_tokens = word_tokenize(code_content)
    corpus = [msg_tokens, code_tokens]

    
    bm25 = BM25Okapi(corpus)
    
    scores = bm25.get_scores(msg_tokens)
    return scores[1]  

def calculate_similarity_lda(msg, code_content):
    """
    使用 LDA 主题模型计算相似度。
    """
    
    texts = [msg.split(), code_content.split()]
    
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]

    
    lda = LdaModel(corpus, num_topics=10, id2word=dictionary, passes=15)

    
    msg_topics = lda.get_document_topics(corpus[0], minimum_probability=0.0)
    code_topics = lda.get_document_topics(corpus[1], minimum_probability=0.0)

    
    msg_vector = [prob for _, prob in msg_topics]
    code_vector = [prob for _, prob in code_topics]
    similarity = cosine_similarity([msg_vector], [code_vector])[0][0]
    return similarity

def calculate_file_similarity(msg, file_content, filepath):
    """
    计算文件变更信息与提交信息的匹配度。
    将文件路径和代码变更内容按照行号顺序组织并与提交信息进行比较。
    """
    added_code = sorted(file_content.get('added_code', {}).items(), key=lambda x: int(x[0]))
    removed_code = sorted(file_content.get('removed_code', {}).items(), key=lambda x: int(x[0]))

    combined_code_lines = [
        f"{added_line} {removed_line}" 
        for (_, added_line), (_, removed_line) in zip(added_code, removed_code)
    ]
    combined_code = " ".join(combined_code_lines)

    file_path_text = filepath.replace('/', ' ').replace('_', ' ').replace('.', ' ').replace('-', ' ')

    full_text = f"{file_path_text} {combined_code}"

    # msg = msg.lower()
    msg_processed = preprocess_text(msg_processed)
    full_text_processed = preprocess_text(full_text)

    return calculate_similarity_jaccard(msg, full_text_processed)


def calculate_similarity(vec1, vec2, method="euclidean", cov_inv=None):
    print(f"*****************排序方法: {method} ******************")
    if method == "cosine":
        return 1 - cosine(vec1, vec2)  
    elif method == "euclidean":
        return -np.linalg.norm(vec1 - vec2)  
    elif method == "mahalanobis":
        if cov_inv is None:
            raise ValueError("计算马氏距离需要协方差矩阵的逆矩阵")
        return -mahalanobis(vec1, vec2, cov_inv)  
    else:
        raise ValueError("Unsupported similarity method")

def calculate_vector_similarity(hash, files, project, method="euclidean", sort="ascending", alpha=0.2, beta=0.8):
    print(f"*****************排序方法: {sort} ******************")

    
    base_path = f"Data_Extraction/git_base/datasets/{project}/"
    csv_path = os.path.join(base_path, "all_similarities.csv")
    
    
    msg_index = faiss.read_index(os.path.join(base_path, "msg_faiss.index"))
    file_index = faiss.read_index(os.path.join(base_path, "file_faiss.index"))

    
    with open(os.path.join(base_path, "faiss_mappings.pkl"), "rb") as f:
        mappings = pickle.load(f)
    msg_map = mappings["msg_map"]
    file_map = mappings["file_map"]

    if hash not in msg_map:
        raise ValueError(f"提交哈希 {hash} 不存在于 FAISS 数据库")
    
    msg_vector = np.zeros(768, dtype=np.float32)
    msg_index.reconstruct(msg_map[hash], msg_vector)

    print(f"*****************相似度计算方法: {method} ******************")
    
    cov_inv = None
    if method == 'mahalanobis':
        all_file_vectors = []
        for file_name in files.keys():
            file_key = (hash, file_name)
            if file_key in file_map:
                file_vector = np.zeros(768, dtype=np.float32)
                file_index.reconstruct(file_map[file_key], file_vector)
                all_file_vectors.append(file_vector)
            else:
                raise ValueError(f"文件 {file_key} 不存在于 FAISS 数据库")
        if method == "mahalanobis" and len(all_file_vectors) > 1:
            cov_matrix = np.cov(np.array(all_file_vectors).T)  
            cov_inv = np.linalg.pinv(cov_matrix)  
        if method == "mahalanobis" and len(all_file_vectors) == 1:
            cov_inv = np.eye(len(all_file_vectors[0]))  

    file_order = {file_name: idx + 1 for idx, file_name in enumerate(files.keys())}
    max_order = max(file_order.values())  

    file_scores = []
    for file_name in files.keys():
        file_key = (hash, file_name)

        if file_key not in file_map:
            raise ValueError(f"文件 {file_key} 不存在于 FAISS 数据库")
        
        file_vector = np.zeros(768, dtype=np.float32)
        file_index.reconstruct(file_map[file_key], file_vector)
        
        similarity = calculate_similarity(msg_vector, file_vector, method, cov_inv)
        
        path_score = 1 - (file_order[file_name] - 1) / (max_order - 1) if max_order > 1 else 1

        final_score = alpha * path_score + beta * similarity
        file_scores.append((file_name, final_score))

    reverse_order = (sort == "descending")  
    file_scores.sort(key=lambda x: x[1], reverse=reverse_order)

    sorted_files = {file_name: files[file_name] for file_name, _ in file_scores}

    write_header = not os.path.exists(csv_path)
    with open(csv_path, mode="a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["commit_hash", "file_name", "score"])
        
        for file_name, score in file_scores:
            writer.writerow([hash, file_name, score])
        print(f"排序结果已追加到 {csv_path}")
    
    return sorted_files