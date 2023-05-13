# -*- coding: utf-8 -*-

import numpy as np

from scipy.sparse import csr_matrix
from scipy.sparse import hstack
from scipy.sparse import vstack

from pyfunctions.general import *
from methods.extraction.general import *
    
def populateLabels(tuples, text, ys, args, stage = False):
    """
    * Given a list of tuples (word, context, label), give a 1 value for 
    * the label if the word matches the actual token label
    """
    for i, tokenTuple in enumerate(tuples):
        center, center_ind, center_types, context, label = tokenTuple
        if not stage:
            if " ".join(center) == text and ys[center_ind] == 1:
                tuples[i] = (center, center_ind, center_types, context, 1)
            else:
                tuples[i] = (center, center_ind, center_types, context, 0)   
        else:
            if text in " ".join(center) and ys[center_ind] == 1 and len(center) == 1:
                tuples[i] = (center, center_ind, center_types, context, 1)
            else:
                tuples[i] = (center, center_ind, center_types, context, 0)   
    return tuples 

def getTuples(document, wordTypes, args):
    """
    * For a document, return a list of tuples (phrase, context, type, label)
    * Initialize the label to be 0 for now
    """
    k = args['k']
    lst = []
    n = len(document)
    i = 0
    while i < n:
        """
        * If token is in body of document
        """
        if i > k and i < n-k:
            for m in range(1,args['max_phrase']):
                context = []
                for j in range(i-k, i):
                    context.append(document[j])
                """
                * Add center phrase
                """
                center, center_types = [], []
                j = i
                while j < i+ m and j < n:
                    center.append(document[j])
                    center_types.append(wordTypes[j])
                    j+=1

                """
                * Add context
                """
                for l in range(j, j+k):
                    if l < n:
                        context.append(document[l])

                lst.append((center, i, center_types, context,0))
        i+=1
    return lst

def tuples2X(tuples, args):
    """
    * Given the data as a list of tuples containing the word, word type,
    * context, and label; return the X features scipy sparse matrix
    """
    X = []
    for i, tokenTuple in enumerate(tuples):
        center, center_ind, center_types, context, label = tokenTuple
        for j, context_word in enumerate(context):
            """
            * Unk word if needed
            """
            wordType = getTokenType(context_word)
            if args['counter'][context_word] < 1:
                if wordType == 1:
                    context[j] = "<unk>"
                elif wordType == 2:
                    context[j] = "<1>"
                else:
                    context[j] = "<$>"
            else:
                context[j] = context_word
        context = " ".join(context)

        for j, center_word in enumerate(center):
            """
            * Unk word if needed
            """
            wordType = getTokenType(center_word)
            if args['counter'][center_word] < 1:
                if wordType == 1:
                    center[j] = "<unk>"
                elif wordType == 2:
                    center[j] = "<1>"
                else:
                    center[j] = "<$>"
            else:
                center[j] = center_word

        center = " ".join(center) 

        """
        * Get bag of Ngrams representation of the context and center
        """
        context_vec = args['vectorizer'].transform([context])
        center_vec = args['vectorizer'].transform([center])

        """
        * Set type to be 0 if token is a word and 1 if numeric value
        """
        type_vec = np.zeros(3)
        for center_type in center_types:
            type_vec[center_type-1] += 1

        """
        * Concatenate bag of Ngrams vector and type value
        """
        vec = hstack([center_vec, context_vec, csr_matrix(type_vec)])
        vec = csr_matrix(vec)
        X.append(vec)
    return vstack(X)
    
def getX(report, args):
    """
    * Given a report, convert it to a sprase matrix and return it
    """
    report = report.split()
    types = getDocumentTokenTypes(report)
    tuples = getTuples(report, types, args)

    """
    * Convert report to string
    """
    X = tuples2X(tuples, args)
    return X
    
    
def getY(report, text, ys, args):
    stage = args['stage']
    """
    * Given a report and a label, extract the y array
    * (binary labels for all tokens in the report)
    """
    report = report.split()
    types = getDocumentTokenTypes(report)
    tuples = getTuples(report, types, args)

    """
    * Convert report to string
    """
    tuples = populateLabels(tuples, text, ys, args, stage)
    y = tuples2Y(tuples)
    return y
    
def tuples2Y(data):
    """
    * Given the data as a list of tuples containing the word, word type,
    * context, and label; return the labels as a numpy array
    """
    y = []
    for j, tuples in enumerate(data):
        center, center_ind, center_types, context, label = tuples
        y.append(label)
    return np.array(y)

def getPhrases(document, args):
    """
    * For a document, return a list of tuples (phrase, context, type, label)
    * Initialize the label to be 0 for now
    """
    document = document.split()
    k = args['k']
    lst = []
    n = len(document)
    i = 0
    while i < n:
        """
        * If token is in body of document
        """
        if i > k and i < n-k:
            for m in range(1, args['max_phrase']):
                context = []
                for j in range(i-k, i):
                    context.append(document[j])
                """
                * Add center phrase
                """
                center = []
                j = i
                while j < i+ m and j < n:
                    center.append(document[j])
                    j+=1
                lst.append((" ".join(center),i))
        i+=1
    return lst


def getX(report, args, m):
    """
    * Given a report, convert it to a sprase matrix and return it
    """
    report = report.split()
    types = getDocumentTokenTypes(report)
    tuples = getTuples(report, types, args, m)

    """
    * Convert report to string
    """
    X = tuples2X(tuples, args)
    return X

def getPhrases(document, args, m):
    """
    * For a document, return a list of tuples (phrase, context, type, label)
    * Initialize the label to be 0 for now
    """
    document = document.split()
    k = args['k']
    lst = []
    n = len(document)
    i = 0
    while i < n:
        """
        * If token is in body of document
        """
        if i > k and i < n-k:
            context = []
            for j in range(i-k, i):
                context.append(document[j])
            """
            * Add center phrase
            """
            center = []
            j = i
            while j < i+ m and j < n:
                center.append(document[j])
                j+=1
            lst.append((" ".join(center),i))
        i+=1
    return lst