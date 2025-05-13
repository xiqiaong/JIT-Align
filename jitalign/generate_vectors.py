import pickle
import faiss
import torch
from transformers import T5Tokenizer, T5Model, RobertaTokenizer, T5Config, T5EncoderModel
import numpy as np
import re
import argparse
import os
import pandas as pd
from tqdm import tqdm


parser = argparse.ArgumentParser(description="Build FAISS index for project")
parser.add_argument("--project", type=str, required=True, help="Project name")
args = parser.parse_args()


project = args.project
output_path = f"../Data_Extraction/git_base/datasets/{project}/"
os.makedirs(output_path, exist_ok=True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = RobertaTokenizer.from_pretrained("../pretrained_model/codet5-base")
# config = T5Config.from_pretrained("../pretrained_model/codet5-base")
model = T5EncoderModel.from_pretrained("../pretrained_model/codet5-base").to(device)
model.eval()

def load_dataset(dataset_path):
    """加载数据集，格式为四元组 (commit_id, files, msg, label)"""
    ddata = pd.read_pickle(dataset_path)
    commit_ids, labels, msgs, codes = ddata
    return list(zip(commit_ids, codes, msgs, labels))

def clean_commit_message(message):
    """
    去除提交消息中以 'change-id' 开头的部分，只保留前面的内容。
    """
    
    cleaned_message = re.split(r"change[-\s]*id", message, flags=re.IGNORECASE)[0].strip()
    return cleaned_message


def encode_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    
    return outputs.last_hidden_state.mean(dim=1).cpu().numpy().squeeze() 


dataset_path = f"../Data_Extraction/git_base/datasets/{project}/jitalign/{project}_all.pkl"

dataset = load_dataset(dataset_path)


dimension = 768
msg_index = faiss.IndexFlatL2(dimension)  
file_index = faiss.IndexFlatL2(dimension) 


msg_map = {}  # commit_id -> msg_index
file_map = {}  # (commit_id, file_name) -> file_index

msg_vectors = []
file_vectors = []
msg_keys = []
file_keys = []


for commit_id, files, msg, _ in tqdm(dataset, desc="Generating vectors"):
    msg = clean_commit_message(msg)
    msg_vector = encode_text(msg)
    msg_map[commit_id] = len(msg_vectors) 
    msg_vectors.append(msg_vector)
    msg_keys.append(commit_id)

    for file_name, changes in files.items():
        added_code = " ".join(v for k, v in sorted(changes.get("added_code", {}).items()))
        removed_code = " ".join(v for k, v in sorted(changes.get("removed_code", {}).items()))
        file_code = f"[ADD] {added_code} [DEL] {removed_code}"
        file_vector = encode_text(file_code)

        file_map[(commit_id, file_name)] = len(file_vectors)  
        file_vectors.append(file_vector)
        file_keys.append((commit_id, file_name))

msg_vectors = np.array(msg_vectors, dtype=np.float32)
file_vectors = np.array(file_vectors, dtype=np.float32)
msg_index.add(msg_vectors)
file_index.add(file_vectors)

faiss.write_index(msg_index, os.path.join(output_path, "msg_faiss.index"))
faiss.write_index(file_index, os.path.join(output_path, "file_faiss.index"))

with open(os.path.join(output_path, "faiss_mappings.pkl"), "wb") as f:
    pickle.dump({"msg_map": msg_map, "file_map": file_map, "msg_keys": msg_keys, "file_keys": file_keys}, f)

print("FAISS 索引构建完成，数据已保存至:", output_path)
