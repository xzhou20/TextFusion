# sentence task
GPU=3
HF_DATASETS_OFFLINE=1
TASK_NAME=sst2
CUDA_VISIBLE_DEVICES=${GPU} python ./train_sentence.py --task_name ${TASK_NAME} --model_name_or_path bert-base-uncased --output_dir ./save_models/sentence/${TASK_NAME}/ft
CUDA_VISIBLE_DEVICES=${GPU} python ./train_sentence_phase1.py --task_name ${TASK_NAME} --model_name_or_path ./save_models/sentence/${TASK_NAME}/ft --output_dir ./save_models/sentence/${TASK_NAME}/phase1
CUDA_VISIBLE_DEVICES=${GPU} python ./train_sentence_phase2.py --task_name ${TASK_NAME} --model_name_or_path ./save_models/sentence/${TASK_NAME}/phase1 --output_dir ./save_models/sentence/${TASK_NAME}/phase2
CUDA_VISIBLE_DEVICES=${GPU} python ./attack_models/attack_distance_model.py --model_name_or_path  ./save_models/sentence/${TASK_NAME}/phase2
CUDA_VISIBLE_DEVICES=${GPU} python ./attack_models/attack_invBert.py --model_name_or_path  ./save_models/sentence/${TASK_NAME}/phase2