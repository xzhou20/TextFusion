import torch
from tqdm import tqdm
import argparse
import os
import sys
import copy

sys.path.append("/root/TextFusion")

from transformers import AdamW, AutoTokenizer, AutoModelForTokenClassification, AutoModelForSequenceClassification,AutoConfig, get_scheduler, set_seed, AutoModelForMaskedLM
from attack_models.attack_metric import *
from data_utils import build_sentence_dataloader, build_token_dataloader


class InversionPLM(torch.nn.Module):
    def __init__(self, config, model_name_or_path='bert-base-uncased'):
        super(InversionPLM, self).__init__()
        self.model = AutoModelForMaskedLM.from_pretrained(model_name_or_path)
        self.loss = torch.nn.CrossEntropyLoss()
    
    def forward(self, x, label, attention_mask=None):
        outputs = self.model(inputs_embeds=x, labels=label, attention_mask=attention_mask)
        return outputs.logits, outputs.loss

    def predict(self, x, label=None, attention_mask=None):
        outputs = self.model(inputs_embeds=x, labels=label, attention_mask=attention_mask)
        logits = outputs.logits
        pred = torch.argmax(torch.nn.functional.softmax(logits,dim=-1), dim=2)
        return logits, pred

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

def word_filter(eval_label, filter_list):
    allow_token_ids = (eval_label == filter_list[0])
    for item in filter_list:
        allow_token_ids = allow_token_ids | (eval_label == item)
    return allow_token_ids

def dataloader2memory(dataloader, model, target_layer=3):
    features = []
    pro_bar = tqdm(range(len(dataloader)))
    model.eval()
    device = model.device
    for batch in dataloader:
        with torch.no_grad():
            batch = {key:value.to(device) for key,value in batch.items()}
            outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], output_hidden_states=True)
            target_hidden_states = outputs['hidden_states'][target_layer].to('cpu')
            target_attention_mask = batch['attention_mask'].to('cpu')
            target_label = batch['input_ids'].to('cpu')
            task_label = batch['labels'].to('cpu')
            features.append({'target_hidden_states': target_hidden_states, 'target_label':target_label, 'target_attention_mask':target_attention_mask, 'task_label':task_label})
        pro_bar.update(1)
    return features

def dataloader2memory_textfusion(dataloader, model, target_layer=3):
    features = []
    pro_bar = tqdm(range(len(dataloader)))
    model.eval()
    device = model.device
    for batch in dataloader:
        with torch.no_grad():
            batch = {key:value.to(device) for key,value in batch.items()}
            outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], output_hidden_states=True)
            target_hidden_states = outputs['hidden_states'][target_layer].to('cpu')
            target_attention_mask = (outputs['overall_info']['temp_attention_mask']==0).squeeze().type_as(batch['attention_mask']).to('cpu')
            if target_hidden_states.shape[:2] != target_attention_mask.shape:
                target_attention_mask = target_attention_mask.unsqueeze(1)
            target_label = batch['input_ids'].to('cpu')
            task_label = batch['labels'].to('cpu')
            features.append({'target_hidden_states': target_hidden_states, 'target_label':target_label, 'target_attention_mask':target_attention_mask, 'task_label':task_label})
        pro_bar.update(1)
    return features
           
def train_inversion_model(train_dataloader, eval_dataloader=None, inversion_epochs=5, inversion_lr=5e-5, device='cuda', output_dir=None):
    inversion_model = InversionPLM(config)
    inversion_model.to(device)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in inversion_model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01,
        },
        {
            "params": [p for n, p in inversion_model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=inversion_lr)

    total_step = len(train_dataloader) * inversion_epochs
    lr_scheduler = get_scheduler(
        name='linear',
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=total_step,
    )

    progress_bar = tqdm(range(total_step))
    
    special_tokens = tokenizer.convert_tokens_to_ids(tokenizer.special_tokens_map.values())
    filter_tokens = list(set(special_tokens))
    
    # device = accelerator.device
    completed_steps = 0
    print('################# start train inversion model #################')
    
    best_attack = 0
    for epoch in range(inversion_epochs):
        for step, batch in enumerate(train_dataloader):            
            batch = {key:value.to(device) for key,value in batch.items()}
            
            target_hidden_states = batch['target_hidden_states']
            target_label = batch['target_label']
            target_attention_mask = batch['target_attention_mask']
            target_label[word_filter(target_label, filter_tokens)]=-100
            logits, loss = inversion_model(target_hidden_states, target_label, attention_mask=target_attention_mask)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            completed_steps += 1
            progress_bar.update(1)
            progress_bar.set_description('loss:{}'.format(loss.item()))                
        torch.save(inversion_model, f'{output_dir}/inversion_model_ft.pt')
        print(f'save inversion model for ft in {output_dir}/inversion_model_ft.pt')
    return inversion_model
                
def evaluate_inversion_model(inversion_model, eval_dataloader, tokenizer, task_type='sentence', device='cuda'):
    special_tokens = [tokenizer.pad_token, tokenizer.cls_token, tokenizer.sep_token]
    metric_list = ['token_hit', 'rouge']
    if task_type == 'token':
        metric_list.append('ent_hit')
    metric_counter = MetricCounter(metric_list)
    
    progress_bar = tqdm(range(len(eval_dataloader)))
    print('################# evaluate inversion model #################')
    attack_results = {}
    for batch in eval_dataloader:
        batch = {key:value.to(device) for key,value in batch.items()}
        target_hidden_states = batch['target_hidden_states']
        target_label = batch['target_label']
        target_attention_mask = batch['target_attention_mask']
        task_label = batch['task_label']
        pred_logits, preds = inversion_model.predict(target_hidden_states, attention_mask=target_attention_mask)
        batch_size = preds.shape[0]
        for i in range(batch_size):
            temp_hit, temp_total = rouge(target_label[i].unsqueeze(0), preds[i].unsqueeze(0), tokenizer, special_tokens)
            metric_counter.update('rouge', temp_hit, temp_total)
            temp_hit, temp_total = token_hit(target_label[i].unsqueeze(0),  preds[i].unsqueeze(0), tokenizer, special_tokens)
            metric_counter.update('token_hit', temp_hit, temp_total)
            if task_type == 'token':
                temp_hit, temp_total = ent_hit(target_label[i].unsqueeze(0), preds[i].unsqueeze(0), tokenizer, special_tokens, label_ids=task_label[i].unsqueeze(0), id2label={idx:item for idx, item in enumerate(label_list) })
                metric_counter.update('ent_hit', temp_hit, temp_total)
        progress_bar.update(1)
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
        "--batch_size",
        type=int,
        default=8,
    )
    # parser.add_argument(
    #     "--task",
    #     type=str,
    #     default='sentence',
    #     choices=['sentence', 'token']
    # )
    args = parser.parse_args()

    model_name_or_path = args.model_name_or_path

    task_info = model_name_or_path.split('/')
    if 'sentence' in task_info:
        args.task = 'sentence'
    else:
        args.task = 'token'
    key_index = task_info.index(args.task)

    # textfusion cape fine-tune dpnr 
    args.dataset_name = task_info[key_index+1]
    device = 'cuda'

    print('########### invBert attack in {} {} #################'.format(args.dataset_name, args.model_name_or_path))

    config = AutoConfig.from_pretrained(model_name_or_path)
    from models.modeling_bert_textfusion_sentence import BertForSequenceClassification as TextFusionBertForSequenceClassification
    from models.modeling_bert_textfusion_token import BertForTokenClassification as TextFusionBertForTokenClassification

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    if args.task == 'sentence':
        model = TextFusionBertForSequenceClassification.from_pretrained(model_name_or_path, config=config).to(device)
        train_dataloader, eval_dataloader = build_sentence_dataloader(args.dataset_name, tokenizer=tokenizer, batch_size=args.batch_size, is_test=True)
    else:
        model = TextFusionBertForTokenClassification.from_pretrained(model_name_or_path, config=config).to(device)
        train_dataloader, eval_dataloader, label_list = build_token_dataloader(args.dataset_name, tokenizer=tokenizer, batch_size=args.batch_size, is_test=True)

    attack_layer = args.target_layer
    
    # train inversion model for fine-tune
    ft_model_path = '/'.join(task_info[:key_index+2]+['ft'])    
    ft_model = AutoModelForTokenClassification.from_pretrained(ft_model_path)
    print('load fine-tune feature to memory')
    ft_train_dataloader = dataloader2memory(train_dataloader, ft_model, target_layer=args.target_layer)
    inversion_model = train_inversion_model(ft_train_dataloader, output_dir=args.model_name_or_path)
    # inversion_model = torch.load(args.model_name_or_path+'/inversion_model_ft.pt')
    
    # evaluate inversion model on textfusion
    print('load textfusion feature to memory')
    textfusion_eval_dataloader = dataloader2memory_textfusion(eval_dataloader, model, target_layer=args.target_layer)
    attack_results = evaluate_inversion_model(inversion_model, eval_dataloader=textfusion_eval_dataloader, tokenizer=tokenizer, task_type=args.task)

    metric_str_list = []
    for key, value in attack_results.items():
        print('layer {} attack {}: {}'.format(attack_layer, key, value))
        metric_str_list += [key, ':', str(value)] 
    metric_str = ' '.join(metric_str_list)

    with open(f'./logs/{args.task}_textfusion.txt', 'a') as f:
        f.write('InvBert-attack task: {}, model_path:{}, {}\n\n'.format(args.dataset_name, args.model_name_or_path, metric_str))



        
        