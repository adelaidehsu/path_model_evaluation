import numpy as np
import scipy

from nltk.tokenize import TreebankWordTokenizer
from pyfunctions.general import extractListFromDic
from scipy.sparse import csr_matrix, vstack
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

def getX(data, N, min_n = 1, vectorizer=None, ret_vec=False, tfidf=False):
    corpus = extractListFromDic(data, 'clean_document_unked')

    lines = []
    for document in corpus:
        document = document.split(" newline ")
        for line in document:
            lines.append(line)

    if vectorizer == None:
        vectorizer = getTrainedVectorizer(lines, N, min_n, tfidf)  

    X = vectorizer.transform(lines)
    if ret_vec:
        return X, vectorizer
    return X

def getXLines(data, field, N, min_n=1, tfidf=False):
    corpus = extractListFromDic(data, 'clean_lines_unked', field)
    corpus = [" newline ".join(document) for document in corpus]

    vectorizer = getTrainedVectorizer(corpus, N, min_n, tfidf)  

    X = vectorizer.transform(corpus)
    return X, vectorizer

def getXLinesFiltered(data, model, N, line_vectorizer, label_vectorizer, min_n=1, prob=False):
    final_X = []
    probs = []
    for i, patient in enumerate(data):
        doc = patient['clean_document_unked']
        lines = doc.split(' newline ')
        Xs = [line_vectorizer.transform([line]) for line in lines]
        pred_lines = [model.predict(X) for X in Xs]
        prob_lines = [model.predict_proba(X)[:,1] for X in Xs]
        probs.append(max(prob_lines))
        filtered_lines = [l for i,l in enumerate(lines) if pred_lines[i] == 1]
        final_X.append(label_vectorizer.transform([" newline ".join(filtered_lines)]))
    if not prob:
        return vstack(final_X)
    return vstack(final_X), probs

def getXLinesFiltered2(data, model, N, line_vectorizer, label_vectorizer, min_n=1, prob=False):
    final_X = []
    probs = []
    for i, patient in enumerate(data):
        doc = patient['clean_document_unked']
        lines = doc.split(' newline ')
        Xs = [line_vectorizer.transform([line]) for line in lines]
        pred_lines = [model.predict(X) for X in Xs]
        prob_lines = [model.predict_proba(X)[:,1] for X in Xs]
        probs.append(max(prob_lines))
        filtered_lines = [l for i,l in enumerate(lines) if pred_lines[i] == 1]
        
        if len(filtered_lines) == 0:
            filtered_lines = [lines[np.argmax(prob_lines)]]
        
        final_X.append(label_vectorizer.transform([" newline ".join(filtered_lines)]))
    if not prob:
        return vstack(final_X)
    return vstack(final_X), probs

def getXTrueLines(data, field, N, label_vectorizer, min_n = 1):
    final_X, final_y = [], []
    for i, patient in enumerate(data):
        final_X.append(label_vectorizer.transform([ 'newline '.join(patient['clean_lines_unked'])]))
        final_y.append(patient['labels'][field])

    return vstack(final_X), final_y

def getTrainedVectorizer(corpus, N, min_n, tf_idf=False):
    if not tf_idf:
        textVectorizer = CountVectorizer(stop_words=None,
                                tokenizer=TreebankWordTokenizer().tokenize, 
                                ngram_range = (min_n,N))
    else:
        textVectorizer = TfidfVectorizer(stop_words=None,
                                tokenizer=TreebankWordTokenizer().tokenize, 
                                ngram_range = (min_n,N))
    textVectorizer.fit(corpus)
    return textVectorizer

def weightFeatures(X_filtered, data, label_vectorizer, weight_method = 'weighted'):
    X_filtered = [X.split(' newline ') for X in X_filtered]
    X_features = [label_vectorizer.transform(X).astype(float) for X in X_filtered]
    for i in range(len(X_features)):
        for j in range(min(len(X_filtered[i]),len(data[i]['filtered_line_probs']))):
            # Weight features by line probability
            if weight_method == 'weighted':
                X_features[i][j] = X_features[i][j]*data[i]['filtered_line_probs'][j]
            else:
                X_features[i][j] = X_features[i][j]
        X_features[i] = csr_matrix(scipy.sparse.csr_matrix.sum(X_features[i], axis=0))
    return X_features