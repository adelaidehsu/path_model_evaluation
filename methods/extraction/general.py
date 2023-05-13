import numpy as np
import random

from collections import Counter
    
type2Val = { 'word':1, 'numeric':2, 'other':3 }
numbers = [str(i) for i in range(10)]
letters = [letter for letter in 'abcdefghijklmnopqrstuvwxyz']


def getTokenType(token):
    """
    * Return the token type given a token (either word or numeric)
    """
    for digit in numbers:
        for letter in letters:
            if digit in token and letter in token:
                return type2Val['other']

    for digit in numbers:
        if digit in token:
            return type2Val['numeric']

    for letter in letters:
        if letter in token:
            return type2Val['word']

    return type2Val['other']

def unkReport(report, args):
    """ 
    * Given a report as a list of strings, unk words that appear 
    * one time or less
    """
    counter = args['counter']
    processed = []
    for token in report.split():
        if counter[token] < 2:
            if getTokenType(token) == 1:
                processed.append("<unk>")
            elif getTokenType(token) == 2:
                processed.append("<1>")
            elif getTokenType(token) == 3:
                processed.append("<1.1>")
            else:
                processed.append("<@>")
        else:
            processed.append(token)
    return " ".join(processed)

def removeShortReports(corpus, labels):
    """
    * Remove reports that may be progress reports (few words) that may not 
    * contain data elements
    """
    new_corpus = []
    new_labels = []
    for i, triple in enumerate(corpus):
        mrn, acc, report = triple
        if len(report.split()) > 50:
            new_corpus.append((mrn, acc, report))
            new_labels.append(labels[i])
    return new_corpus, new_labels

def getCounter(corpus_train):
    """
    * Given a list of report as strings, return a Python counter on the 
    * concatenated reports
    """
    lst = []
    for doc in corpus_train:
        lst = lst + doc.split()
    counter = Counter(lst)
    return counter

def getDocumentTokenTypes(document):
    """
    * Given a document, return a list of the token types that appear
    """
    types = []
    for token in document:
        token = token.lower()
        tokenType = getTokenType(token)
        types.append(tokenType)
    return types


def sampleTrain(X, y, args):
    """
    * Upsample training instances
    """
    sample = args['sample']
    positive_indices = np.where(y == 1)[0]
    negative_indices = np.where(y == 0)[0]
    train_negative = np.random.choice(negative_indices, sample).tolist()
    train_positive = np.random.choice(positive_indices, sample).tolist()
    train_indices = train_positive + train_negative
    random.shuffle(train_indices)
    return X[train_indices], y[train_indices]

