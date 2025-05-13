# JIT-Align
JIT-Align: A Semantic Alignment-Based Ranking Framework for Just-In-Time Defect Prediction  

2025 lEEE 49th Annual Computers, Software, and Applications Conference (COMPSAC)

## Datasets: MC4Defect
Get Link: https://drive.google.com/drive/folders/1NvtwTEo-CKNQ2TgyMU5nxeJ6m7YmaylA?usp=drive_link
## Run JITAlign
1. Construction of the vector FAISS database
   Run generate_vectors.py for the corresponding project.
   
   For example:
   
   `run generate_vectors.py --project openstack`
   
   The corresponding folder generates the corresponding `msg_faiss.index`, `file_faiss.index` and
`faiss_mappings.pkl` files.
3. run JITAlign
   Semantic alignment is already included in the run, note the optimal parameter settings: `alpha = 0.2`, `beta = 0.8`.
   
   For example:
   ```
   python -m jitalign.semantic.run --output_dir=model/jitalign/openstack/saved_models_semantic/checkpoints --config_name=pretrained_model/codet5-base --model_name_or_path=pretrained_model/codet5-base --tokenizer_name=pretrained_model/codet5-base --do_train --train_data_file Data_Extraction/git_base/datasets/openstack/jitalign/openstack_train.pkl --eval_data_file Data_Extraction/git_base/datasets/openstack/jitalign/openstack_val.pkl --test_data_file Data_Extraction/git_base/datasets/openstack/jitalign/openstack_test.pkl --epoch 30 --max_seq_length 512 --max_msg_length 128 --train_batch_size 24 --eval_batch_size 32 --learning_rate 1e-5 --max_grad_norm 1.0 --evaluate_during_training --patience 10 --code_sequence_mode=vector_sort --code_sequence_settings=descending --seed 42 2>&1| tee model/simcom/openstack/saved_models_semantic/train.log 
   ```
   ```
   python -m jitalign.semantic.run --output_dir=model/jitalign/openstack/saved_models_semantic/checkpoints --config_name=pretrained_model/codet5-base --model_name_or_path=pretrained_model/codet5-base --tokenizer_name=pretrained_model/codet5-base --do_test --train_data_file Data_Extraction/git_base/datasets/openstack/jitalign/openstack_train.pkl --eval_data_file Data_Extraction/git_base/datasets/openstack/jitalign/openstack_val.pkl --test_data_file Data_Extraction/git_base/datasets/openstack/jitalign/openstack_test.pkl --epoch 30 --max_seq_length 512 --max_msg_length 128 --train_batch_size 24 --eval_batch_size 100 --code_sequence_mode=vector_sort --code_sequence_settings=descending --learning_rate 1e-5 --max_grad_norm 1.0 --patience 10 --seed 42 2>&1| tee model/simcom/openstack/saved_models_semantic/test.log 
   ```
   
## Supplementary experimental results
### Discussion1: Impact of similarity algorithms on model performance.
![image](https://github.com/user-attachments/assets/4c31d201-5624-423a-8143-bb1b12e31322)

### Discussion2: Impact of Parameter Settings in Semantic Alignment Algorithms on Model Performance
![image](https://github.com/user-attachments/assets/92630973-1e06-4d3a-840b-9a908c9982a9)

If you have any questions, please feel free to share and discuss :smile:.



## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

All images and content © 2025 Yujie Ye.



