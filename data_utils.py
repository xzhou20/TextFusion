import os
import datasets
from transformers import DataCollatorWithPadding, DataCollatorForTokenClassification
from torch.utils.data import DataLoader

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

def convert_conll_to_json():
    pass

def build_sentence_dataloader(dataset_name='sst2', tokenizer=None, padding=False, max_length=128, batch_size=8, is_test=False):
    if dataset_name in task_to_keys.keys():
        raw_datasets = datasets.load_dataset("glue", dataset_name)
    else:
        raw_datasets = datasets.load_dataset(dataset_name)
    raw_datasets.pop('test')

    label_list = raw_datasets["train"].unique("label")
    label_list.sort()
    num_labels = len(label_list)
    label_to_id = {v: i for i, v in enumerate(label_list)}

    sentence1_key, sentence2_key = task_to_keys[dataset_name]
    def preprocess_function_sentence(examples):
        # Tokenize the texts
        texts = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*texts, padding=padding, max_length=max_length, truncation=True)

        if "label" in examples:
            if label_to_id is not None:
                # Map labels to IDs (not necessary for GLUE tasks)
                result["labels"] = [label_to_id[l] for l in examples["label"]]
            else:
                # In all cases, rename the column to labels because the model will expect that.
                result["labels"] = examples["label"]
        return result
    processed_datasets = raw_datasets.map(
            preprocess_function_sentence,
            batched=True,
            remove_columns=raw_datasets["train"].column_names,
            desc="Running tokenizer on dataset",
            load_from_cache_file=False
        )
    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["validation"]
    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=None)
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=batch_size
    )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=batch_size)
    return train_dataloader, eval_dataloader

def build_token_dataloader(dataset_name='conll2003', tokenizer=None, padding=False, max_length=128, batch_size=8, label_all_tokens=False, is_test=False):
    raw_datasets = datasets.load_dataset(dataset_name)
    text_column_name = "tokens"

    column_names = raw_datasets["train"].column_names
    features = raw_datasets["train"].features
    if "ner_tags" in column_names:
        label_column_name =  "ner_tags"
    else:
        label_column_name = column_names[1]

    def get_label_list(labels):
        unique_labels = set()
        for label in labels:
            unique_labels = unique_labels | set(label)
        label_list = list(unique_labels)
        label_list.sort()
        return label_list
    labels_are_int = isinstance(features[label_column_name].feature, datasets.ClassLabel)
    if labels_are_int:
        label_list = features[label_column_name].feature.names
        label_to_id = {i: i for i in range(len(label_list))}
    else:
        label_list = get_label_list(raw_datasets["train"][label_column_name])
        label_to_id = {l: i for i, l in enumerate(label_list)}

    b_to_i_label = []
    for idx, label in enumerate(label_list):
        if label.startswith("B-") and label.replace("B-", "I-") in label_list:
            b_to_i_label.append(label_list.index(label.replace("B-", "I-")))
        else:
            b_to_i_label.append(idx)


    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples[text_column_name],
            max_length=max_length,
            padding=padding,
            truncation=True,
            # We use this argument because the texts in our dataset are lists of words (with a label for each word).
            is_split_into_words=True
        )
        
        labels = []
        for i, label in enumerate(examples[label_column_name]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(label_to_id[label[word_idx]])
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    if label_all_tokens:
                        label_ids.append(b_to_i_label[label_to_id[label[word_idx]]])
                    else:
                        label_ids.append(-100)
                previous_word_idx = word_idx

            labels.append(label_ids)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    processed_raw_datasets = raw_datasets.map(
            tokenize_and_align_labels,
            batched=True,
            remove_columns=raw_datasets["train"].column_names,
            desc="Running tokenizer on dataset",
            load_from_cache_file=False
        )

    train_dataset = processed_raw_datasets["train"]
    if 'validation' in processed_raw_datasets and not is_test:
        eval_dataset = processed_raw_datasets["validation"]
    else:
        eval_dataset = processed_raw_datasets["test"]

    data_collator = DataCollatorForTokenClassification(
            tokenizer, pad_to_multiple_of=None
        )
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=batch_size
    )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=batch_size)
    return train_dataloader, eval_dataloader, label_list

