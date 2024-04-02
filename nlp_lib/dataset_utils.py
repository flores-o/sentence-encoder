from torch.utils.data import DataLoader, TensorDataset
import torch
from torch.utils.data import DataLoader
import gzip
import csv
import pandas as pd
from transformers import BertTokenizer

from IPython.display import display, HTML


def log_info(message, verbose=True):
    if verbose:
        display(HTML(f"<span style='color: yellow'>{message}</span>"))

########## Utility functions for STS dataset #########


def load_sts_dataset(file_name, verbose=False):
    # Read the CSV file into a pandas DataFrame
    data = pd.read_csv(file_name, compression='gzip',
                       delimiter='\t', quoting=csv.QUOTE_NONE)

    log_info(
        f"load_sts_dataset start _________________________________", verbose=verbose)

    # Sanity checks:
    for col in ['split', 'sentence1', 'sentence2', 'score']:
        if col not in data.columns:
            raise ValueError(f"Column '{col}' not found in the dataset.")
    if 'test' not in data['split'].unique():
        raise ValueError("Test split not found in the dataset.")

    # Extract test data from the DataFrame
    test_data = data[data['split'] == 'test'][[
        'sentence1', 'sentence2', 'score']]

    # Log details about the test data
    log_info(
        f"Extracted test data with {test_data.shape[0]} rows. Preview of the test data:", verbose=verbose)

    # Ensure each sample has both sentences.
    sts_samples = {'test': [(row['sentence1'], row['sentence2'], row['score'])
                            for _, row in test_data.iterrows()]}

    # Log a few samples to verify
    log_info(f"Sample data from sts_samples:", verbose=verbose)
    for sample in sts_samples['test'][:2]:
        log_info(
            f"Sentence1: {sample[0]}, Sentence2: {sample[1]}, Score: {sample[2]}")

    log_info(f"load_sts_dataset end _________________________________",
             verbose=verbose)
    return sts_samples


def tokenize_sentence_pair_sts_dataset(dataset, tokenizer: BertTokenizer, max_length=512, verbose=False):
    # Sanity checks:
    if not dataset:
        raise ValueError("The provided dataset is empty.")
    if tokenizer is None:
        raise ValueError("Tokenizer is not provided or is None.")

    log_info(
        f"Starting tokenization for {len(dataset)} sentence pairs.", verbose=verbose)

    # Tokenize each sentence pair individually
    tokenized_dataset = [tokenizer(s[0], s[1],
                                   padding='max_length',
                                   truncation=True,
                                   max_length=max_length,
                                   return_tensors="pt",
                                   add_special_tokens=True) for s in dataset]

    # Log details about the first tokenized pair as an example
    example_tokenized = tokenized_dataset[0]
    log_info(f"First sentence pair {dataset[0]}", verbose=verbose)
    log_info(f"Sample tokenized data for first sentence pair:", verbose=verbose)
    log_info(f"Input IDs: {example_tokenized['input_ids']}", verbose=verbose)
    log_info(
        f"Attention Mask: {example_tokenized['attention_mask']}", verbose=verbose)
    log_info(
        f"Token Type IDs: {example_tokenized['token_type_ids']}", verbose=verbose)

    dataset_for_loading = []
    for i, tokenized in enumerate(tokenized_dataset):
        entry = {
            'input_ids': tokenized['input_ids'].squeeze(0),
            'attention_mask': tokenized['attention_mask'].squeeze(0),
            'token_type_ids': tokenized['token_type_ids'].squeeze(0),
            'score': torch.tensor(dataset[i][2], dtype=torch.float)
        }
        dataset_for_loading.append(entry)

    # Logging the shapes
    log_info(
        f"Shape of input_ids for a sample: {dataset_for_loading[0]['input_ids'].shape}", verbose=verbose)
    log_info(
        f"Shape of attention_mask for a sample: {dataset_for_loading[0]['attention_mask'].shape}", verbose=verbose)
    log_info(
        f"Shape of token_type_ids for a sample: {dataset_for_loading[0]['token_type_ids'].shape}", verbose=verbose)
    log_info(
        f"Type and value of score for a sample: {type(dataset_for_loading[0]['score'])}, {dataset_for_loading[0]['score']}", verbose=verbose)

    return dataset_for_loading


def collate_fn(batch):
    # Sanity checks:
    if not batch:
        raise ValueError(
            "Batch is empty. Ensure DataLoader is working correctly.")

    # Stack everything together
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    token_type_ids = torch.stack([item['token_type_ids'] for item in batch])
    scores = torch.stack([item['score'] for item in batch])

    verbose = False
    log_info(
        "collate_fn start_________________________________________", verbose=verbose)
    log_info(f"input_ids shape: {input_ids.shape}", verbose=verbose)
    log_info(f"attention_mask shape: {attention_mask.shape}", verbose=verbose)
    log_info(f"token_type_ids shape: {token_type_ids.shape}", verbose=verbose)
    log_info(f"scores shape: {scores.shape}", verbose=verbose)
    log_info(
        "collate_fn end _________________________________________", verbose=verbose)

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'token_type_ids': token_type_ids,
        'score': scores
    }


def get_sts_dataloader(tokenized_dataset, batch_size, shuffle=False):
    return DataLoader(tokenized_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)

########## Utility functions for STS dataset - V2 #########


def tokenize_sentence_pair_sts_dataset_v2(dataset, tokenizer, max_length=512, verbose=False):
    tokenized_dataset = []

    for s1, s2, score in dataset:
        tokenized_1 = tokenizer(s1, padding='max_length', truncation=True,
                                max_length=max_length, return_tensors="pt", add_special_tokens=True)
        tokenized_2 = tokenizer(s2, padding='max_length', truncation=True,
                                max_length=max_length, return_tensors="pt", add_special_tokens=True)

        entry = {
            'input_ids_1': tokenized_1['input_ids'].squeeze(0),
            'attention_mask_1': tokenized_1['attention_mask'].squeeze(0),
            'input_ids_2': tokenized_2['input_ids'].squeeze(0),
            'attention_mask_2': tokenized_2['attention_mask'].squeeze(0),
            'score': torch.tensor(score, dtype=torch.float)
        }
        tokenized_dataset.append(entry)

    return tokenized_dataset


def collate_fn_v2(batch):
    # Sanity check:
    if not batch:
        raise ValueError(
            "Batch is empty. Ensure DataLoader is working correctly.")

    # Stack input_ids and attention_mask for both sentences in all items in the batch.
    input_ids_1 = torch.stack([item['input_ids_1'] for item in batch])
    attention_mask_1 = torch.stack(
        [item['attention_mask_1'] for item in batch])
    input_ids_2 = torch.stack([item['input_ids_2'] for item in batch])
    attention_mask_2 = torch.stack(
        [item['attention_mask_2'] for item in batch])

    # Stack the scores for all items in the batch.
    scores = torch.stack([item['score'] for item in batch])

    # Prepare the return dictionary. Include input_ids and attention_mask for both sentences, and scores.
    ret = {
        'input_ids_1': input_ids_1,
        'attention_mask_1': attention_mask_1,
        'input_ids_2': input_ids_2,
        'attention_mask_2': attention_mask_2,
        'score': scores
    }

    # Check if 'token_type_ids' exists in the first item of the batch for both sentences.
    # If 'token_type_ids' is present for any sentence, stack it.
    # This handles models like BERT which use token_type_ids.
    if 'token_type_ids_1' in batch[0]:
        token_type_ids_1 = torch.stack(
            [item['token_type_ids_1'] for item in batch])
        ret['token_type_ids_1'] = token_type_ids_1

    if 'token_type_ids_2' in batch[0]:
        token_type_ids_2 = torch.stack(
            [item['token_type_ids_2'] for item in batch])
        ret['token_type_ids_2'] = token_type_ids_2

    return ret


def get_sts_dataloader_v2(tokenized_dataset, batch_size, shuffle=False):
    return DataLoader(tokenized_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn_v2)
######### Utility functions for NLI dataset ############


def load_nli_dataset(file_name):
    data = pd.read_csv(file_name, compression='gzip',
                       delimiter='\t', quoting=csv.QUOTE_NONE)
    nli_samples = {'train': data[data['split'] == 'train'][[
        'sentence1', 'sentence2', 'label']].values.tolist()}
    return nli_samples

# To do - add DEVICE to config.py file and pass it as default parameter


def tokenize_sentence_pair_nli_dataset(dataset, tokenizer, device, max_length=128):
    sentence1_list = [item[0] for item in dataset]
    sentence2_list = [item[1] for item in dataset]
    labels_list = [item[2] for item in dataset]

    # Convert labels to numeric values: {'contradiction': 0, 'entailment': 1, 'neutral': 2}
    label_map = {'contradiction': 0, 'entailment': 1, 'neutral': 2}
    labels = [label_map[label] for label in labels_list]

    tokenized_data = tokenizer(sentence1_list, sentence2_list, padding='max_length',
                               truncation=True, max_length=max_length, return_tensors='pt')
    tokenized_data['labels'] = torch.tensor(labels).to(device)

    return tokenized_data


def get_nli_dataloader(tokenized_data, batch_size=8, shuffle=True):
    dataset = TensorDataset(tokenized_data["input_ids"],
                            tokenized_data["attention_mask"],
                            tokenized_data.get("token_type_ids", torch.zeros_like(
                                tokenized_data["input_ids"])),
                            tokenized_data["labels"])

    dataloader = DataLoader(dataset, shuffle=shuffle, batch_size=batch_size)
    return dataloader


# Used with BertContrastive model
def adjust_labels_for_contrastive_loss(tokenized_data):
    mask = (tokenized_data['labels'] != 2)

    tokenized_data['input_ids'] = tokenized_data['input_ids'][mask]
    tokenized_data['attention_mask'] = tokenized_data['attention_mask'][mask]
    tokenized_data['token_type_ids'] = tokenized_data['token_type_ids'][mask]
    tokenized_data['labels'] = tokenized_data['labels'][mask]
    tokenized_data['labels'][tokenized_data['labels'] == 1] = 1  # entailment
    tokenized_data['labels'][tokenized_data['labels']
                             == 0] = 0  # contradiction
    return tokenized_data
