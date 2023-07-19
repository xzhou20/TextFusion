from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
import torch

def ent_hit(input_ids, pred_ids, tokenizer, special_tokens, label_ids, id2label):
    hit_cnt = 0
    total_cnt = 0
    for i in range(input_ids.shape[0]):
        real_tokens = tokenizer.convert_ids_to_tokens(input_ids[i])
        pred_sentence = tokenizer.decode(pred_ids[i], skip_special_tokens=True)
        # pred_sentence = pred_sentence.replace('Baghdad', '')
        golden_labels = label_ids[i]
        
        ent_list = []
        ent = []
        for label, token in zip(golden_labels, real_tokens):
            label = label.item()
            if token in special_tokens:
                continue
            if label == -100:
                label = 'I'
            else:
                label = id2label[label]
            if label == 'O':
                if ent:
                    ent_list.append(ent)
                    ent = []
            elif label[0] == 'B':
                if ent:
                    ent_list.append(ent)
                    ent = []
                ent.append(token)
            else:
                if ent:
                    ent.append(token)
        if ent:
            ent_list.append(ent)
        
        for ent in ent_list:
            ent = tokenizer.convert_tokens_to_string(ent)
            if ent in pred_sentence:
                hit_cnt += 1
        total_cnt += len(ent_list)
    return hit_cnt, total_cnt

def token_hit_multi_label(input_ids, pred_ids, tokenizer, special_tokens):
    batch_real_tokens = [tokenizer.convert_ids_to_tokens(item) for item in input_ids]
    batch_pred_tokens = [tokenizer.convert_ids_to_tokens(torch.round(item).nonzero()) for item in pred_ids]
    # batch_real_tokens = [tokenizer.decode(item, skip_special_tokens=True).split() for item in input_ids]
    # batch_pred_tokens = [tokenizer.decode(item, skip_special_tokens=True).split() for item in pred_ids]
    hit_cnt = 0
    total_cnt = 0
    for real_tokens, pred_tokens in zip(batch_real_tokens, batch_pred_tokens):
        real_tokens = list(set(real_tokens))
        pred_tokens = list(set(pred_tokens))
        real_tokens = [token.lower() for token in real_tokens if token not in special_tokens]
        pred_tokens = [token.lower() for token in pred_tokens if token not in special_tokens]
        for token in real_tokens:
            if token in special_tokens:
                continue
            if token in pred_tokens:
                hit_cnt += 1
            total_cnt += 1
    return hit_cnt, total_cnt

def token_hit(input_ids, pred_ids, tokenizer, special_tokens):
    batch_real_tokens = [tokenizer.convert_ids_to_tokens(item) for item in input_ids]
    batch_pred_tokens = [tokenizer.convert_ids_to_tokens(item) for item in pred_ids]
    # batch_real_tokens = [tokenizer.decode(item, skip_special_tokens=True).split() for item in input_ids]
    # batch_pred_tokens = [tokenizer.decode(item, skip_special_tokens=True).split() for item in pred_ids]
    hit_cnt = 0
    total_cnt = 0
    # "it's", 'a', 'charming', 'and', 'often', 'affecting', 'journey.'
    for real_tokens, pred_tokens in zip(batch_real_tokens, batch_pred_tokens):
        real_tokens = list(set(real_tokens))
        pred_tokens = list(set(pred_tokens))
        real_tokens = [token.lower() for token in real_tokens if token not in special_tokens]
        pred_tokens = [token.lower() for token in pred_tokens if token not in special_tokens]
        for token in real_tokens:
            if token in special_tokens:
                continue
            if token in pred_tokens:
                hit_cnt += 1
            total_cnt += 1
    return hit_cnt, total_cnt

def bleu4(input_ids, pred_ids, tokenizer, special_tokens, n_gram=4):
    batch_real_tokens = [tokenizer.decode(item, skip_special_tokens=True).split() for item in input_ids]
    # batch_real_tokens = [tokenizer.convert_ids_to_tokens(item, skip_special_tokens=True) for item in input_ids]
    batch_pred_tokens = [tokenizer.decode(item, skip_special_tokens=True).split() for item in pred_ids]
    # batch_pred_tokens = [tokenizer.convert_ids_to_tokens(item, skip_special_tokens=True) for item in pred_ids]
    
    hit_cnt = 0
    total_cnt = 0
    gram_weight = tuple([1/n_gram]*n_gram)
    for real_tokens, pred_tokens in zip(batch_real_tokens, batch_pred_tokens):
        # pred_tokens = [token.lower() for token in pred_tokens if token not in special_tokens]
        # real_tokens = [token.lower() for token in real_tokens]
        bleu_score = sentence_bleu([real_tokens], pred_tokens, gram_weight)
        hit_cnt += bleu_score
        total_cnt += 1
    return hit_cnt, total_cnt

def rouge(input_ids, pred_ids, tokenizer, special_tokens):
    batch_real_tokens = [tokenizer.decode(item, skip_special_tokens=True) for item in input_ids]
    batch_pred_tokens = [tokenizer.decode(item, skip_special_tokens=True) for item in pred_ids]

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    # scores = scorer.score('The quick brown fox jumps over the lazy dog',
    #                   'The quick brown dog jumps on the log.')
    hit_cnt = 0
    total_cnt = 0
    for real_tokens, pred_tokens in zip(batch_real_tokens, batch_pred_tokens):
        rouge_score = scorer.score(real_tokens, pred_tokens)['rougeL'].fmeasure
        hit_cnt += rouge_score
        total_cnt += 1
    return hit_cnt, total_cnt
    
