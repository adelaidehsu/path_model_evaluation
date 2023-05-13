import copy
import numpy as np
import torch

from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader, TensorDataset
from pyfunctions.general import *

def getVocab(data):
    """
     * Given a list of documents, get all the unique vocabulary
    """
    vocab = []
    for patient in data:
        document = patient['clean_document'].split()
        for word in document:
            if word not in vocab:
                vocab.append(word)
    return vocab

def encodeLabels(data, encoder, field):
    for i, patient in enumerate(data):
        encoded = encoder.transform([str(patient['labels'][field])])[0]
        if 'encoded_labels' in patient:
            patient['encoded_labels'][field] = encoded
        else:
            patient['encoded_labels'] = {field: encoded}
        data[i] = patient
    return data

def getEncoder(data, field):
    train = []
    for patient in data:
        train.append(str(patient['labels'][field]))
    encoder = LabelEncoder()
    encoder.fit(train)
    return encoder

def getTorchLoader(corpus, labels, args, shuffle):        
    features = torch.zeros(len(corpus), args['maxDocLength'], dtype = torch.long)

    for i, doc in enumerate(corpus):
        doc = doc.split()
        if len(doc) > args['maxDocLength']:
            doc = doc[0:args['maxDocLength']]

        docVec = torch.zeros(args['maxDocLength'], dtype = torch.long)

        j = args['maxDocLength'] - 1
        for word in list(reversed(doc)):
            docVec[j] = args['word2idx'][word]
            j-=1

        features[i,:] = docVec

    targets = torch.LongTensor(labels)
    dataset = TensorDataset(features, targets)

    loader = DataLoader(dataset, batch_size= args['batchSize'], shuffle=shuffle, num_workers=0)
    return loader

def get_data(corpus, labels, args, shuffle):        
    features = torch.zeros(len(corpus), args['maxDocLength'], dtype = torch.long)

    for i, doc in enumerate(corpus):
        doc = doc.split()
        if len(doc) > args['maxDocLength']:
            doc = doc[0:args['maxDocLength']]

        docVec = torch.zeros(args['maxDocLength'], dtype = torch.long)

        j = args['maxDocLength'] - 1
        for word in list(reversed(doc)):
            docVec[j] = args['word2idx'][word]
            j-=1

        features[i,:] = docVec

    targets = torch.LongTensor(labels)
    return features, targets

def reSample(corpus, labels):
    """
    * Given reports and labels, upsample minority classes so
    * each class has an equal number of instances and return expanded data as a list
    """
    corpus_lst = corpus

    labels_lst = copy.deepcopy(labels)
    """
    * Get the maximum number a class appears in the labels
    """
    max_occur = getNumMaxOccurrences(labels)

    """
    * Get list containing each class index set (indices where a class shows up)
    """
    classIndices = getClassIndices(np.array(labels))

    """
    * Loop through each class index set
    """
    for indices in classIndices:
        """
        * Get subset that matches current class
        """
        subLabels = np.array(labels)[indices]
        subCorpus = [data for i, data in enumerate(corpus) if i in indices ]

        """
        * Upsample the class so that the number of instances matches max_occur
        """
        n = len(subLabels)
        upsampled_indices = np.random.choice(range(n), size=max_occur - n, replace=True)
        if len(upsampled_indices) > 0:
            upsampled_labels = subLabels[upsampled_indices]
            upsampled_corpus = [subCorpus[ind] for ind in upsampled_indices]

            """
            * Add upsampled data to report and labels lists
            """
            corpus_lst = corpus_lst + upsampled_corpus
            labels_lst = labels_lst + upsampled_labels.tolist()
    return corpus_lst, labels_lst

class TNLRFeaturesDataset(Dataset):

    def __init__(self, features, labels, doc_length, embedding_dim, transform=None):
        self.features = features
        self.labels = labels
        self.doc_length = doc_length
        self.embedding_dim = embedding_dim
        self.n = len(features)
        self.classes = len(np.unique(labels))
        
        self.padded_features = []
        
        for feature in self.features:
            padded = torch.zeros(self.doc_length, self.embedding_dim, dtype=torch.float)
            len_ = min(self.doc_length, feature.shape[1])
            padded[:len_, :] = torch.tensor(feature[0,:len_,:], dtype=torch.float)
            
            self.padded_features.append(padded)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):        
        return self.padded_features[idx], self.labels[idx]

def make_weights_for_balanced_classes(labels, nclasses):                        
    count = [0] * nclasses                                                      
    for label in labels:                                                         
        count[label] += 1                                                     
    weight_per_class = [0.] * nclasses                                      
    N = float(sum(count))                                                   
    for i in range(nclasses):
        if count[i] == 0:
            weight_per_class[i] = 0
        else:
            weight_per_class[i] = N/float(count[i])                                 
    weight = [0] * len(labels)                                              
    for idx, label in enumerate(labels):                                          
        weight[idx] = weight_per_class[label]                                  
    return weight

def TNLRFeatureLoader(features, labels, args, shuffle):        
    targets = torch.LongTensor(labels)
    dataset = TNLRFeaturesDataset(features, targets, args['maxDocLength'], args['embeddingDim'])
    
    if shuffle:
        weights = make_weights_for_balanced_classes(labels, args['nclasses'])                                                                
        weights = torch.DoubleTensor(weights)                                       
        sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))                     
                                                                                    
        loader = DataLoader(dataset, batch_size=args['batchSize'], sampler=sampler, num_workers=0, pin_memory=True)
        
    else:
        loader = DataLoader(dataset, batch_size=args['batchSize'], shuffle=shuffle, num_workers=0, pin_memory=True)
    return loader

def TNLRFeatureLoader2(features, labels, args, shuffle):        
    targets = torch.LongTensor(labels)
    dataset = TNLRFeaturesDataset(features, targets, args['maxDocLength'], args['embeddingDim'])
    
    if shuffle:
        weights = make_weights_for_balanced_classes(labels, args['nclasses'])                                                                
        weights = torch.DoubleTensor(weights)                                       
        sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))                     
                                                                                    
        loader = DataLoader(dataset, batch_size=args['batchSize'], sampler=sampler, num_workers=0, pin_memory=False)
        
    else:
        loader = DataLoader(dataset, batch_size=args['batchSize'], shuffle=shuffle, num_workers=0, pin_memory=False)
    return loader