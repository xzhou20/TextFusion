# sentence task
GPU=2
HF_DATASETS_OFFLINE=1

# token task
TASK_NAME=conll2003
CUDA_VISIBLE_DEVICES=${GPU} python ./train_token.py --dataset_name ${TASK_NAME} --model_name_or_path bert-base-uncased --output_dir ./save_models/token/${TASK_NAME}/ft
CUDA_VISIBLE_DEVICES=${GPU} python ./train_token_phase1.py --dataset_name ${TASK_NAME} --model_name_or_path ./save_models/token/${TASK_NAME}/ft --output_dir ./save_models/token/${TASK_NAME}/phase1
CUDA_VISIBLE_DEVICES=${GPU} python ./train_token_phase2.py --dataset_name ${TASK_NAME} --model_name_or_path ./save_models/token/${TASK_NAME}/phase1 --output_dir ./save_models/token/${TASK_NAME}/phase2
CUDA_VISIBLE_DEVICES=${GPU} python ./attack_models/attack_distance_model.py --model_name_or_path  ./save_models/token/${TASK_NAME}/phase2
CUDA_VISIBLE_DEVICES=${GPU} python ./attack_models/attack_invBert.py  --model_name_or_path  ./save_models/token/${TASK_NAME}/phase2