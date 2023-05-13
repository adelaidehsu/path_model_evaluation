import argparse
import numpy as np
import os
import pandas as pd
import sys
import torch
import torch.nn.functional as F
import warnings
import pickle

warnings.filterwarnings("ignore")

base_dir = os.path.split(os.getcwd())[0]
sys.path.append(base_dir)
sys.path.append(f"{base_dir}/turing/examples-raw/gluesst_finetune/")
sys.path.append(f"{base_dir}/turing/src/")

from argparse import Namespace
from methods.bag_of_ngrams.processing import cleanReports, cleanSplit, stripChars
from methods.interpretations.utils import compute_input_type_attention
from methods.interpretations.integrated_gradients.utils import forward_with_softmax, summarize_attributions
from pyfunctions.general import extractListFromDic, readJson
from pyfunctions.pathology import extract_synoptic, fixLabelProstateGleason, fixProstateLabels, fixLabel, exclude_labels
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from transformers import AutoTokenizer, AutoModel
from transformers import BertTokenizer, BertForSequenceClassification
from turing.pathology.path_utils import evaluate, extract_features, load_tnlr_base, load_tnlr_tokenizer, path_dataset

args_dic = {
    'do_train': False,
    'do_eval': True,
    'evaluate_during_training': True,
    'max_seq_length': 512,
    'do_lower_case': True,
    'per_gpu_train_batch_size': 8,
    'per_gpu_eval_batch_size': 8,
    'gradient_accumulation_steps': 1,
    'learning_rate': 7e-6,
    'weight_decay': 0.0,
    'adam_epsilon': 1e-8,
    'max_grad_norm': 1,
    'num_train_epochs': 3.0,
    'max_steps': -1,
    'warmup_ratio': 0.2,
    'logging_steps': 50,
    'eval_all_checkpoints': True,
    'no_cuda': False,
    'seed': 42,
    'metric_for_choose_best_checkpoint': None,
    'fp16': False,
    'fp16_opt_level': 'O1',
    'local_rank': -1,
    'num_train_epochs': 25, 
    'n_gpu': 1,
    'device': 'cuda',
    'run': 0
}

parser = argparse.ArgumentParser()
parser.add_argument("-model_type",type=str)
parser.add_argument("-task",type=str)
parser.add_argument("-all",type=str)
parser.add_argument("-layer_num",type=int)
parser.add_argument("-epoch_num",type=int)

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
    tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract")
elif args['model_type'] == 'biobert':
    bert_path = "dmis-lab/biobert-v1.1"
    tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-v1.1")
elif args['model_type'] == 'clinical_biobert':
    bert_path = "emilyalsentzer/Bio_ClinicalBERT"
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
elif args['model_type'] == 'tnlr':
    vocab_file = f'{base_dir}/turing/src/tnlr/tokenizer/tnlr-uncased-vocab.txt'
    tokenizer = load_tnlr_tokenizer(vocab_file)

def featurize(dataloader, model, flag='ft'):
    collect = []
    for batch in tqdm(dataloader, desc=f"Featurizing {flag}"):
        model.eval()
        batch = tuple(t.to(args['device']) for t in batch)

        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "labels": batch[3],
            }
            if args['model_type'] != "distilbert":
                inputs["token_type_ids"] = (
                    batch[2] if args['model_type'] in ["bert", "xlnet"] else None
                )  # XLM, DistilBERT and RoBERTa don't use segment_ids

            outputs = model(**inputs)
            if args['model_type'] != 'tnlr':
                layer_outputs = outputs[2] #loss, logits, hidden, attn
            else:
                layer_outputs = outputs[2][0] #loss, logits, hidden  
        layer_outputs = torch.stack(layer_outputs) #[layer, batch, seq, d]
        cls_layer_outputs = torch.tensor(layer_outputs[:, :, 0, :]) #[layer, batch, d]
        collect.append(cls_layer_outputs)

    feature = torch.cat(collect, dim=1) #[layer, N, d]
    return feature

# Read in data
path = f"../data/prostate.json"
data = readJson(path)

# Clean reports
data = cleanSplit(data, stripChars)
data['dev_test'] = cleanReports(data['dev_test'], stripChars)
data = fixLabel(data)

fields = [args['task']]

for field in fields:
    # Create datasets
    train_documents = [extract_synoptic(patient['document'].lower(), tokenizer) for patient in data['train']]
    train_labels = [patient['labels'][field] for patient in data['train']]

    val_documents = [extract_synoptic(patient['document'].lower(), tokenizer) for patient in data['val']]
    val_labels = [patient['labels'][field] for patient in data['val']]

    test_documents = [extract_synoptic(patient['document'].lower(), tokenizer) for patient in data['test']]
    test_labels = [patient['labels'][field] for patient in data['test']]
    
    # Exclude '2' and 'null'
    if field in ['PrimaryGleason', 'SecondaryGleason']:
        train_documents, train_labels = exclude_labels(train_documents, train_labels)
        val_documents, val_labels = exclude_labels(val_documents, val_labels)
        test_documents, test_labels = exclude_labels(test_documents, test_labels)

    le = preprocessing.LabelEncoder()
    le.fit(train_labels)

    # Handle new class
    le_dict = dict(zip(le.classes_, le.transform(le.classes_)))
    le_dict = {str(key):le_dict[key] for key in le_dict}
    
    for label in val_labels + test_labels:
        if str(label) not in le_dict:
            le_dict[str(label)] = len(le_dict)
            
    train_labels = [le_dict[str(label)] for label in train_labels]
    val_labels = [le_dict[str(label)] for label in val_labels]
    test_labels = [le_dict[str(label)] for label in test_labels]

    # Map processed label back to raw label
    inv_le_dict = {v: k for k, v in le_dict.items()}

    documents_full = train_documents + val_documents + test_documents
    labels_full = train_labels + val_labels + test_labels

    p_test = len(test_labels)/len(labels_full)
    p_val = len(val_labels)/(len(train_labels) + len(val_labels))
    
   
    ft_model_path = f"{base_dir}/output/fine_tuning/{args['model_type']}_{args['run']}/{field}"

    if args['epoch_num']:
        #specify snapshot
        ft_checkpoint_file = f"{ft_model_path}/epoch-{args['epoch_num']}"
        ft_config_file = f"{ft_model_path}/epoch-{args['epoch_num']}/config.json"
    else:
        #best epoch
        ft_checkpoint_file = f"{ft_model_path}/save_output"
        ft_config_file = f"{ft_model_path}/save_output/config.json"
        

    if args['model_type'] != 'tnlr':
        ft_model = BertForSequenceClassification.from_pretrained(ft_checkpoint_file, num_labels=len(le_dict), output_hidden_states=True)
    else:
        ft_model = load_tnlr_base(ft_checkpoint_file, ft_config_file, model_type='tnlrv3_classification', num_labels=len(le_dict))
        ft_model.config.update({'output_hidden_states': True})
    
    train_docs, test_docs, train_labels, test_labels = train_test_split(documents_full, 
                                                                        labels_full, 
                                                                        test_size= p_test,
                                                                        random_state=args['run'])

    train_docs, val_docs, train_labels, val_labels = train_test_split(train_docs, 
                                                                      train_labels, 
                                                                      test_size= p_val,
                                                                      random_state=args['run'])
               
    if args['all']:
        samples = test_docs
        labels = test_labels
    else:
        samples = test_docs
        labels = test_labels

    ft_dataset = path_dataset(samples, labels, ft_model, tokenizer)
    ft_dataloader = DataLoader(ft_dataset, batch_size=args['per_gpu_train_batch_size'])
    
    ft_model = ft_model.to(args['device'])
    ft_layer_cls_logits1 = featurize(ft_dataloader, ft_model, flag='ft') #[layer, N, d]
    
    if args['all']:
        comb_collect_ft_layer_outputs = torch.tensor(ft_layer_cls_logits1).detach().cpu().numpy()
    else:
        comb_collect_ft_layer_outputs = torch.tensor(ft_layer_cls_logits1[args['layer_num']]).detach().cpu().numpy()
    
    # save last layer features
    save_path = f"{base_dir}/output/rsa/{args['model_type']}/{field}"
    os.makedirs(save_path, exist_ok=True)
    
    if args['epoch_num']:
        with open(os.path.join(save_path, f"{len(samples)}_cls_logits_ft_epoch{args['epoch_num']}.pkl"), 'wb') as handle:
            pickle.dump(comb_collect_ft_layer_outputs, handle)
    else:
        with open(os.path.join(save_path, f"{len(samples)}_cls_logits_l{args['layer_num']}_ft_best.pkl"), 'wb') as handle:
            pickle.dump(comb_collect_ft_layer_outputs, handle)
        
    
    del ft_model
    del ft_dataset
    del ft_dataloader