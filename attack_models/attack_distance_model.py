import torch
import tqdm
import argparse
import os
import sys
import copy

sys.path.append("/root/TextFusion")

from transformers import AdamW, AutoTokenizer, AutoModelForTokenClassification, AutoConfig, get_scheduler, set_seed, AutoModelForMaskedLM
from attack_models.attack_metric import *
from data_utils import build_sentence_dataloader, build_token_dataloader

class MetricCounter():
    def __init__(self, keys=None):
        self.hit = dict([(key,0) for key in keys])
        self.total = dict([(key,0) for key in keys])
        

    def update(self, key=None, hit=None, total=None):
        self.hit[key] += hit
        self.total[key] += total

    def __call__(self,key=None):
        if self.total[key]!=0:
            return self.hit[key] / self.total[key]
        else:
            return 0

def knn_attack(victim_model, tokenizer, eval_dataloader, label_list, attack_layer=3, emb=None, is_token=False, topk=1):
    device = victim_model.device
    special_tokens = [tokenizer.pad_token, tokenizer.cls_token, tokenizer.sep_token]
    metric_list = ['token_hit', 'rouge']
    if is_token:
        metric_list.append('ent_hit')
    metric_counter = MetricCounter(metric_list)
    if emb == None:
        emb = copy.deepcopy(victim_model.bert.embeddings.word_embeddings.weight)
    for batch in tqdm.tqdm(eval_dataloader):
        batch = {key:value.to(device) for key,value in batch.items()}
        labels = batch.pop('labels')
        batch['output_hidden_states'] = True
        with torch.no_grad():
            outputs = victim_model(**batch)
        masks = batch['attention_mask'].tolist()
        batch_size = batch['input_ids'].shape[0]

        for i in range(batch_size):
            seq_len = len(masks[i]) if 0 not in masks[i] else masks[i].index(0)
            hidden_states = outputs.hidden_states[attack_layer][i]

            ed = torch.cdist(hidden_states, emb, p=2.0)
            candidate_token_ids_topk = torch.topk(ed,topk,largest=False)[1]

            temp_hit, temp_total = rouge(batch['input_ids'][i].unsqueeze(0), candidate_token_ids_topk[:,0].unsqueeze(0), tokenizer, special_tokens)
            metric_counter.update('rouge', temp_hit, temp_total)
            temp_hit, temp_total = token_hit(batch['input_ids'][i].unsqueeze(0), candidate_token_ids_topk[:,0].unsqueeze(0), tokenizer, special_tokens)
            metric_counter.update('token_hit', temp_hit, temp_total)
            if is_token:
                temp_hit, temp_total = ent_hit(batch['input_ids'][i].unsqueeze(0), candidate_token_ids_topk[:,0].unsqueeze(0), tokenizer, special_tokens, label_ids=labels[i].unsqueeze(0), id2label={idx:item for idx, item in enumerate(label_list) })
                metric_counter.update('ent_hit', temp_hit, temp_total)
    attack_results = {metric: metric_counter(metric) for metric in metric_list}
    return attack_results




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default='/root/TextFusion/save_models/token/conll2003/phase2/5e-05/epoch_3',
    )
    parser.add_argument(
        "--target_layer",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--task",
        type=str,
        default='sentence',
        choices=['sentence', 'token']
    )
    args = parser.parse_args()

    model_name_or_path = args.model_name_or_path

    task_info = model_name_or_path.split('/')
    if 'sentence' in task_info:
        args.task = 'sentence'
        is_token = False
    else:
        args.task = 'token'
        is_token = True
    key_index = task_info.index(args.task)

    # textfusion cape fine-tune dpnr 
    args.dataset_name = task_info[key_index+1]
    device = 'cuda'

    print('########### KNN attack in {} {} #################'.format(args.dataset_name, args.model_name_or_path))

    config = AutoConfig.from_pretrained(model_name_or_path)
    from models.modeling_bert_textfusion_sentence import BertForSequenceClassification
    from models.modeling_bert_textfusion_token import BertForTokenClassification

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    if args.task == 'sentence':
        model = BertForSequenceClassification.from_pretrained(model_name_or_path, config=config).to(device)
        train_dataloader, eval_dataloader = build_sentence_dataloader(args.dataset_name, tokenizer=tokenizer, batch_size=args.batch_size, is_test=True)
        label_list = None
    else:
        model = BertForTokenClassification.from_pretrained(model_name_or_path, config=config).to(device)
        train_dataloader, eval_dataloader, label_list = build_token_dataloader(args.dataset_name, tokenizer=tokenizer, batch_size=args.batch_size, is_test=True)

    attack_results = knn_attack(model, tokenizer=tokenizer, eval_dataloader=eval_dataloader, label_list=label_list, attack_layer=args.target_layer, is_token=is_token)
    
    metric_str_list = []
    for key, value in attack_results.items():
        print('layer {} attack {}: {}'.format(args.target_layer, key, value))
        metric_str_list += [key, ':', str(value)] 
    metric_str = ' '.join(metric_str_list)

    with open(f'./logs/{args.task}_textfusion.txt', 'a') as f:
        f.write('KNN-attack task: {}, model_path:{}, {}\n\n'.format(args.dataset_name, args.model_name_or_path, metric_str))



        
        