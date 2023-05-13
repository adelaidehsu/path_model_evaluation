import argparse
import numpy as np
import os
import pandas as pd
import sys
import torch
import warnings

warnings.filterwarnings("ignore")

base_dir = os.path.split(os.getcwd())[0]
sys.path.append(base_dir)

sys.path.append(f"{base_dir}/turing/examples-raw/gluesst_finetune/")
sys.path.append(f"{base_dir}/turing/src/")

from argparse import Namespace
from methods.bag_of_ngrams.processing import cleanReports, cleanSplit, stripChars
from pyfunctions.general import extractListFromDic, readJson
from pyfunctions.pathology import extract_synoptic, fixProstateLabels, fixLabel, exclude_labels
from sklearn import preprocessing
from sklearn.metrics import f1_score
from turing.pathology.run_classifier import MODEL_CLASSES, processors, train_path
from methods.torch.processing import make_weights_for_balanced_classes
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from tqdm import tqdm, trange
from transformers import AutoTokenizer, BertTokenizer, BertForSequenceClassification
from turing.pathology.path_utils import extract_features, load_tnlr_base, load_tnlr_tokenizer, path_dataset
from utils_for_glue import glue_compute_metrics as compute_metrics
from sklearn.model_selection import train_test_split

args_dic = {
    'model_name_or_path': '',
    'task_name': 'sst-2',
    'config_name': '',
    'tokenizer_name': '',
    'do_train': True,
    'do_eval': True,
    'evaluate_during_training': True,
    'max_seq_length': 512,
    'do_lower_case': True,
    'per_gpu_train_batch_size': 8,
    'per_gpu_eval_batch_size': 8,
    'gradient_accumulation_steps': 1,
    'learning_rate': 7.6e-6,
    'weight_decay': 0.01,
    'adam_epsilon': 1e-8,
    'max_grad_norm': 1,
    'max_steps': -1,
    'warmup_ratio': 0.2,
    'logging_steps': 50,
    'eval_all_checkpoints': True,
    'no_cuda': False,
    'overwrite_output_dir': True,
    'seed': 42,
    'overwrite_cache': True,
    'metric_for_choose_best_checkpoint': None,
    'fp16': False,
    'fp16_opt_level': 'O1',
    'local_rank': -1,
    'num_train_epochs': 25,
    'n_gpu': 1,
    'device': 'cuda'
}

parser = argparse.ArgumentParser()
parser.add_argument("-model_type",type=str)
parser.add_argument("-task",type=str)
parser.add_argument("-run",type=int)

args = vars(parser.parse_args())

def Merge(dict1, dict2):
    res = {**dict1, **dict2}
    return res

args = Merge(args, args_dic)

if args['model_type'] == 'bert':
    bert_path = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
elif args['model_type'] == 'pubmed_bert':
    bert_path = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
    tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract", local_files_only=False)
elif args['model_type'] == 'biobert':
    bert_path = "dmis-lab/biobert-v1.1"
    tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-v1.1", local_files_only=False)
elif args['model_type'] == 'clinical_biobert':
    bert_path = "emilyalsentzer/Bio_ClinicalBERT"
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT", local_files_only=False)
elif args['model_type'] == 'tnlr':
    checkpoint_file = f'{base_dir}/turing/src/tnlr/checkpoints/tnlrv3-base.pt'
    config_file = f'{base_dir}/turing/src/tnlr/config/tnlr-base-uncased-config.json'
    vocab_file = f'{base_dir}/turing/src/tnlr/tokenizer/tnlr-uncased-vocab.txt'
    tokenizer = load_tnlr_tokenizer(vocab_file)

# Read in data
path = f"{base_dir}/data/prostate.json"
data = readJson(path)

# Clean reports
data = cleanSplit(data, stripChars)
data['dev_test'] = cleanReports(data['dev_test'], stripChars)

data = fixLabel(data)

fields = [args['task']]

for field in fields:
    # Set output directories
    args['output_dir'] = f"{base_dir}/output/fine_tuning/{args['model_type']}_{args['run']}/{field}"
    args['cache_dir'] = f"{base_dir}/output/fine_tuning/{args['model_type']}_{args['run']}/{field}"
    
    kwargs = Namespace(**args)

    # Create datasets
    train_documents = [extract_synoptic(patient['document'].lower(), tokenizer) for patient in data['train']]
    train_labels = [patient['labels'][field] for patient in data['train']]

    val_documents = [extract_synoptic(patient['document'].lower(), tokenizer) for patient in data['val']]
    val_labels = [patient['labels'][field] for patient in data['val']]

    test_documents = [extract_synoptic(patient['document'].lower(), tokenizer) for patient in data['test']]
    test_labels = [patient['labels'][field] for patient in data['test']]
    
    if field in ['PrimaryGleason', 'SecondaryGleason']:
        train_documents, train_labels = exclude_labels(train_documents, train_labels)
        val_documents, val_labels = exclude_labels(val_documents, val_labels)
        test_documents, test_labels = exclude_labels(test_documents, test_labels)

    le = preprocessing.LabelEncoder()
    le.fit(train_labels)

    # Handle new class
    le_dict = dict(zip(le.classes_, le.transform(le.classes_)))
    le_dict = {str(key):le_dict[key] for key in le_dict}
    print(le_dict)
    for label in val_labels + test_labels:
        if str(label) not in le_dict:
            le_dict[str(label)] = len(le_dict)

    train_labels = [le_dict[str(label)] for label in train_labels]
    val_labels = [le_dict[str(label)] for label in val_labels]
    test_labels = [le_dict[str(label)] for label in test_labels]

    # Train model
    if args['model_type'] != 'tnlr':
        model = BertForSequenceClassification.from_pretrained(bert_path, num_labels=len(le_dict))
    else:
        model = load_tnlr_base(checkpoint_file, config_file, model_type='tnlrv3_classification', num_labels=len(le_dict))
      
    documents_full = train_documents + val_documents + test_documents
    labels_full = train_labels + val_labels + test_labels

    p_test = len(test_labels)/len(labels_full)
    p_val = len(val_labels)/(len(train_labels) + len(val_labels))
    
    
    train_documents, test_documents, train_labels, test_labels = train_test_split(documents_full, 
                                                                                  labels_full, 
                                                                                  test_size= p_test,
                                                                                  random_state=args['run'])

    train_documents, val_documents, train_labels, val_labels = train_test_split(train_documents, 
                                                                                  train_labels, 
                                                                                  test_size= p_val,
                                                                                  random_state=args['run'])

        
    class_weights = make_weights_for_balanced_classes(train_labels, len(train_labels))
    
    train_dataset = path_dataset(train_documents, train_labels, model, tokenizer)
    val_dataset = path_dataset(val_documents, val_labels, model, tokenizer)
    test_dataset = path_dataset(test_documents, test_labels, model, tokenizer)
    
    train_path(kwargs, train_dataset, val_dataset, model, tokenizer, class_weights)
    
    del model
