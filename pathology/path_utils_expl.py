# coding=utf-8
from __future__ import absolute_import, division, print_function

import argparse
import copy
import csv
import glob
import json
import logging
import os
import random
import sys
import json
import shutil

code_dir = os.path.split(os.getcwd())[0]

sys.path.append(f"{code_dir}/turing/src/")

import numpy as np
import torch

from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler

try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

from tqdm import tqdm, trange

from transformers import WEIGHTS_NAME

from transformers import AdamW, get_linear_schedule_with_warmup
from tnlr.modeling_expl import TuringNLRv3ForSequenceClassification, TuringNLRv3Model
from tnlr.configuration_tnlrv3 import TuringNLRv3Config
from tnlr.tokenization_tnlrv3 import TuringNLRv3Tokenizer

from turing.pathology.utils import pathology_convert_examples_to_features as convert_examples_to_features
from turing.pathology.utils import pathology_processors as processors
from utils_for_glue import glue_compute_metrics as compute_metrics

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    "tnlrv3": (
        TuringNLRv3Config,
        TuringNLRv3Model,
        TuringNLRv3Tokenizer,
    ),
    'tnlrv3_classification': (
        TuringNLRv3Config,
        TuringNLRv3ForSequenceClassification,
        TuringNLRv3Tokenizer,
    ),
}

device = torch.device("cpu")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def load_tnlr_base(checkpoint_file, config_file, model_type='tnlrv3', num_labels=None, seed=0):
    processor = processors['prostate']()

    # Set seed
    set_seed(seed)

    config_class, model_class, tokenizer_class = MODEL_CLASSES[model_type]
    config = config_class.from_pretrained(config_file, cache_dir=None, output_hidden_states=True)
    
    if num_labels is not None:
        config.num_labels = num_labels
        
    model = model_class.from_pretrained(
        checkpoint_file, 
        from_tf=bool(".ckpt" in checkpoint_file), config=config, 
        cache_dir=None)

    model.to(device)
    return model

def load_tnlr_tokenizer(vocab_file, model_type='tnlrv3', do_lower_case=True):
    config_class, model_class, tokenizer_class = MODEL_CLASSES[model_type]
    return tokenizer_class.from_pretrained(vocab_file, do_lower_case=do_lower_case,cache_dir=None)


def split_document_into_chunks(document, tokenizer, chunk_size=512):
    """Split long document into chunks of 512 tokens"""
    
    # Make room for [CLS] and [SEP] tokens
    chunk_size = chunk_size-2
    
    encoded_tokens = tokenizer.encode(document)
    n = len(encoded_tokens)
    i = 0
    chunks = []
    
    while i < n:
        i+=chunk_size
        chunks.append(encoded_tokens[(i-chunk_size):i])
    
    for i in range(len(chunks)):
        chunks[i] = tokenizer.decode(chunks[i])
        
    chunks[0] = chunks[0].replace('[CLS] ', '')
    chunks[-1] = chunks[-1].replace(' [SEP]', '')
    
    return chunks


def build_loader(features):
    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor(
        [f.attention_mask for f in features], dtype=torch.long
    )
    all_token_type_ids = torch.tensor(
        [f.token_type_ids for f in features], dtype=torch.long
    )

    all_labels = torch.tensor([f.label for f in features], dtype=torch.float)

    dataset = TensorDataset(
        all_input_ids, all_attention_mask, all_token_type_ids, all_labels
    )

    sampler = SequentialSampler(dataset)
    return DataLoader(dataset, sampler=sampler, batch_size=1)

def compute_chunk_lengths(chunks, tokenizer):
    lengths = []
    n = len(chunks)
    for i, chunk in enumerate(chunks):
        # Note this is including the added [CLS] and [SEP] tokens
        lengths.append(len(tokenizer.encode(chunk)))
    return lengths

    
def path_dataset(documents, labels, model, tokenizer, model_type='tnlrv3', max_seq_length=512):
    
    processor = processors['prostate']()
    #label_list = processor.get_labels()
    label_list = list(set(labels))
    
    examples = (processor.get_examples(documents, labels))

    features = convert_examples_to_features(
            examples,
            tokenizer,
            label_list=label_list,
            max_length=max_seq_length,
            pad_on_left=bool(model_type in ["xlnet"]),  # pad on the left for xlnet
            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            pad_token_segment_id=4 if model_type in ["xlnet"] else 0,
    )
    
    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor(
        [f.attention_mask for f in features], dtype=torch.long
    )
    all_token_type_ids = torch.tensor(
        [f.token_type_ids for f in features], dtype=torch.long
    )
    
    all_labels = torch.tensor([f.label for f in features], dtype=torch.long)

    dataset = TensorDataset(
        all_input_ids, all_attention_mask, all_token_type_ids, all_labels
    )
    return dataset

def extract_features(document, model, tokenizer, model_type='tnlrv3', 
                     max_seq_length=512, method='chunks', second_to_last_layer=False):
    
    document = tokenizer.encode(document)[1:-1] # Remove CLS and SEP tokens
    document = tokenizer.decode(document)
    
    if method == 'chunks':
        return extract_features_chunks(document, model, tokenizer, 
                                       model_type=model_type, max_seq_length=max_seq_length, second_to_last_layer=second_to_last_layer)
    elif method == 'normal':
        return extract_features_normal(document, model, tokenizer, 
                                       model_type=model_type, max_seq_length=max_seq_length, second_to_last_layer=second_to_last_layer)
    elif method == 'sliding window':
        return extract_features_sliding_window(document, model, tokenizer,
                                               model_type=model_type, max_seq_length=max_seq_length, second_to_last_layer=second_to_last_layer)
    else:
        raise ValueError('Not a valid feature extraction method')
        
def extract_features_chunks(document, model, tokenizer, model_type='tnlrv3', max_seq_length=512, second_to_last_layer=False):
    processor = processors['prostate']()
    label_list = processor.get_labels()
    
    document_chunks = split_document_into_chunks(document, tokenizer)
    labels = [0 for _ in range(len(document_chunks))] # hack
    examples_chunks = (processor.get_examples(document_chunks, labels))
    lengths_chunks = compute_chunk_lengths(document_chunks, tokenizer) # includes added [CLS] and [SEP] tokens to each chunk

    features_chunks = convert_examples_to_features(
            examples_chunks,
            tokenizer,
            label_list=label_list,
            max_length=max_seq_length,
            pad_on_left=bool(model_type in ["xlnet"]),  # pad on the left for xlnet
            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            pad_token_segment_id=4 if model_type in ["xlnet"] else 0,
    )

    dataloader = build_loader(features_chunks)

    encoded_features = []
    i = 0
    n_chunks = len(document_chunks)
    for batch in tqdm(dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(device) for t in batch)

        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
            }
            outputs = model(**inputs)
            
            if second_to_last_layer:
                if i == 0 and i == (n_chunks-1):
                    encoded = outputs[-1][0][-2][:,:lengths_chunks[i],:]
                elif i == 0:
                    # Remove [SEP] token since its not the last chunk but keep [CLS] token
                    encoded = outputs[-1][0][-2][:,:(lengths_chunks[i]-1),:]
                elif i == (n_chunks-1):
                    # Remove [CLS] token since its not the first chunk but keep [SEP] token
                    encoded = outputs[-1][0][-2][:,1:(lengths_chunks[i]),:]
                else:
                    # Remove [CLS] token and [SEP] token since its not the first or last chunk
                    encoded = outputs[-1][0][-2][:,1:(1+lengths_chunks[i]-2),:]
            else:
                if i == 0 and i == (n_chunks-1):
                    encoded = outputs[0][:,:lengths_chunks[i],:]
                elif i == 0:
                    # Remove [SEP] token since its not the last chunk but keep [CLS] token
                    encoded = outputs[0][:,:(lengths_chunks[i]-1),:]
                elif i == (n_chunks-1):
                    # Remove [CLS] token since its not the first chunk but keep [SEP] token
                    encoded = outputs[0][:,1:(lengths_chunks[i]),:]
                else:
                    # Remove [CLS] token and [SEP] token since its not the first or last chunk
                    encoded = outputs[0][:,1:(1+lengths_chunks[i]-2),:]
                
            encoded_features.append(encoded)
        i+=1

    return torch.cat(encoded_features, dim=1)

def extract_features_normal(document, model, tokenizer, model_type='tnlrv3', max_seq_length=512, second_to_last_layer=False):
    processor = processors['prostate']()
    label_list = processor.get_labels()
    
    examples = (processor.get_examples([document], [0]))

    features = convert_examples_to_features(
        examples,
        tokenizer,
        label_list=label_list,
        max_length=max_seq_length,
        pad_on_left=bool(model_type in ["xlnet"]),  # pad on the left for xlnet
        pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
        pad_token_segment_id=4 if model_type in ["xlnet"] else 0,
    )

    dataloader = build_loader(features)

    i = 0
    for batch in tqdm(dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(device) for t in batch)

        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
            }
            outputs = model(**inputs)
            
            if second_to_last_layer:
                return outputs[-1][0][-2]
            else:
                return outputs[0]
            
def split_document_into_windows(document, tokenizer, window_size=512):
    """Split long document into windows of 512 tokens"""
    encoded_tokens = tokenizer.encode(document)
    n = len(encoded_tokens)

    windows = []
    
    window_size = window_size - 2
    sliding_step_size = window_size//2
    
    i = 0
    while i < n:
        windows.append((encoded_tokens[i:min(i+window_size, n)], [j for j in range(i, min(i+window_size, n))]))
        i+=sliding_step_size
        
    cs = []
    
    for i in range(len(windows)):
        windows[i] = (tokenizer.decode(windows[i][0]), windows[i][1])

    return windows

def extract_embeddings_with_most_context(embeddings, indices, lengths):
    right_bound = indices[-1][-1] + 1 # Last token of last window inclusive
    
    index_to_context_size = {i:[] for i in range(right_bound)}
    
    for i in range(right_bound):
        
        for j, inds in enumerate(indices):
            try:
                position = inds.index(i)
                max_context_size = min(position, len(inds) - position)
                index_to_context_size[i].append((max_context_size, j, position))
            except:
                continue

    extracted_embeddings = [] 
    
    for i in range(right_bound):
        maximum = max([tuple_[0] for tuple_ in index_to_context_size[i]])
        
        for tuple_ in index_to_context_size[i]:
            if maximum == tuple_[0]:
                extracted_embeddings.append(embeddings[tuple_[1]][:,tuple_[2],:])
                break
    return extracted_embeddings 

def extract_features_sliding_window(document, model, tokenizer, model_type='tnlrv3', max_seq_length=512, second_to_last_layer=False):
    processor = processors['prostate']()
    label_list = processor.get_labels()
    
    windows = split_document_into_windows(document, tokenizer)  
    
    windows, indices = [window[0] for window in windows], [window[1] for window in windows]
    labels = [0 for _ in range(len(windows))] # hack
    examples = (processor.get_examples(windows, labels))
    lengths = compute_chunk_lengths(windows, tokenizer)

    features = convert_examples_to_features(
            examples,
            tokenizer,
            label_list=label_list,
            max_length=max_seq_length,
            pad_on_left=bool(model_type in ["xlnet"]),  # pad on the left for xlnet
            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            pad_token_segment_id=4 if model_type in ["xlnet"] else 0,
    )

    dataloader = build_loader(features)

    encoded_features = []

    for batch in tqdm(dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(device) for t in batch)

        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
            }
            outputs = model(**inputs)
            if second_to_last_layer:
                encoded_features.append(outputs[-1][0][-2])
            else:
                encoded_features.append(outputs[0])
            
    embeddings = extract_embeddings_with_most_context(encoded_features, indices, lengths)
    
    # Convert list of embeddings to tensor
    dim2, dim3 = len(embeddings), embeddings[0].shape[-1]
    return torch.cat(embeddings, dim=1).reshape((1, dim2, dim3))

def evaluate(eval_dataloader, args, model, tokenizer, prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task_names = (
        ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
    )

    results = {}

    # Eval!
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "labels": batch[3],
            }
            if args.model_type != "distilbert":
                inputs["token_type_ids"] = (
                    batch[2] if args.model_type in ["bert", "xlnet"] else None
                )  # XLM, DistilBERT and RoBERTa don't use segment_ids
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]

            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs["labels"].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(
                out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0
            )

    eval_loss = eval_loss / nb_eval_steps
    preds = np.argmax(preds, axis=1)
    result = compute_metrics('mnli', preds, out_label_ids)
    results.update(result)

    results = {'macro': np.round(f1_score(out_label_ids, preds, average='macro'), 3),
               'micro': np.round(f1_score(out_label_ids, preds, average='micro'), 3)}

    return results, out_label_ids, preds