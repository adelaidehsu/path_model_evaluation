import copy
import numpy as np

from collections import Counter
from methods.bag_of_ngrams.processing import cleanReport, stripChars, unkReport
from pyfunctions.general import createDirectory, extractListFromDic, getClassIndices, getNumMaxOccurrences, hasLetter, hasNumeric, listFilesType, readJson, saveJson
from scipy.sparse import csr_matrix, hstack, vstack

def is_synoptic(line):
    stripped_line = line.strip()
    if '-' == stripped_line[0]:
        return True
    else:
        return False

def flattenLines(data,reduce = False):
    for i, patient in enumerate(data):
        for key in patient['lines']:
            flattened_lines = []
            if reduce == True:
                synoptic_lines = []
                first_lines = []
                found_line = False
                found_syn_line = False
                for line in patient['lines'][key]:
                    if len(line) > 0:
                        full_line = line
                        line = line.split('NEWLINE')
                        for subline in line:
                            if is_synoptic(full_line) and found_syn_line == False:
                                synoptic_lines.append(subline)
                            if not found_line:
                                flattened_lines.append(subline)
                        if is_synoptic(full_line):
                            found_syn_line = True
                        if len(flattened_lines) > 0:
                            found_line = True

                            
                if len(synoptic_lines) > 0:
                    flattened_lines = synoptic_lines
                    
                     
            else:    
                for line in patient['lines'][key]:
                    if len(line) > 0:
                        line = line.split('NEWLINE')
                        for subline in line:
                            flattened_lines.append(subline)
            patient['lines'][key] = flattened_lines
        data[i] = patient
    return data


def cleanLines(data):
    for i, patient in enumerate(data):
        patient['clean_lines'] = dict()
        for key in patient['lines']:
            clean_lines = []
            for line in patient['lines'][key]:
                clean_lines.append(cleanReport(line, stripChars + ['-']))
            patient['clean_lines'][key] = clean_lines
        data[i] = patient
    return data

def unkLines(data, counter):
    for i, patient in enumerate(data):
        patient['clean_lines_unked'] = dict()
        for key in patient['clean_lines']:
            unked_lines = []
            for line in patient['clean_lines'][key]:
                unked_lines.append(unkReport(line, counter))
            patient['clean_lines_unked'][key] = unked_lines
        data[i] = patient
    return data


def addLineLabels(data,field_names):
    for i, patient in enumerate(data):
        document_lines = patient['clean_document'].split(" newline ")
        line_labels = {field: [] for field in field_names}
        
        for field in field_names:
            
            for line in document_lines:                
                if line in patient['clean_lines'][field]:
                    line_labels[field].append(1)
                else:
                    line_labels[field].append(0)
        patient['line_labels'] = line_labels
        data[i] = patient  
    return data

def filterLinesWithRules(document,document_unked,vectorizer, k, field, keys):
    lines = document.split(' newline ')
    lines_unked = document_unked.split(' newline ')
    vectors = vectorizer.transform(lines_unked)
    
    filtered_lines = []
    filtered_lines_unked = []
    filtered_vects =  []
    inds = []
    for i,unked_line in enumerate(lines_unked):
        found_key = False
        for key in keys[field]:
            if key in unked_line:
                found_key = True
        if found_key:
            filtered_lines.append(lines[i])
            filtered_lines_unked.append(lines_unked[i])
            filtered_vects.append(vectors[i])
            inds.append(i)
    return ' newline '.join(filtered_lines[:k]), ' newline '.join(filtered_lines_unked[:k]), filtered_vects[:k], inds[:k]
    
def filterLines(document,document_unked, model, vectorizer, k, full_doc = False, sep=' newline '):
    lines = document.split(' newline ')
    lines_unked = document_unked.split(' newline ')
    vectors = vectorizer.transform(lines_unked)
    preds = model.predict(vectors)
    probs = model.predict_proba(vectors)[:,1]
        
    filtered_lines = lines
    filtered_lines_unked = lines_unked
    filtered_vects =  vectors
    
    inds = np.argsort(probs)[::-1]
    top_k = inds[:k]

    if not full_doc:
        filtered_lines = [filtered_lines[i] for i in top_k]
        filtered_lines_unked = [filtered_lines_unked[i] for i in top_k]
        filtered_vects = [filtered_vects[i] for i in top_k]
        filtered_probs = [probs[i] for i in top_k]
        
        return sep.join(filtered_lines), sep.join(filtered_lines_unked), filtered_vects, top_k, filtered_probs
    else:
        filtered_lines = [filtered_lines[i] for i in inds]
        filtered_lines_unked = [filtered_lines_unked[i] for i in inds]
        filtered_vects = [filtered_vects[i] for i in inds]
        filtered_probs = [probs[i] for i in inds]
        return sep.join(filtered_lines), sep.join(filtered_lines_unked), filtered_vects, inds, filtered_probs
    
def getNALines(data, field, label_vectorizer):
    X, y=[],[]
    for patient in data:
        doc = patient['clean_document_unked'].split(' newline ')
        for line in doc:
            if line not in patient['clean_lines_unked'][field]:
                X.append(label_vectorizer.transform([line]))
                y.append("NA")
    return vstack(X),y

def getTrueLines(data, field, N, label_vectorizer, min_n = 1):
    final_X, final_y = [], []
    for i, patient in enumerate(data):
        for line in patient['clean_lines_unked'][field]:
            final_X.append(label_vectorizer.transform([line]))
            final_y.append(patient['labels'][field])

    return vstack(final_X), final_y
    
def joinAdjacentLines(data, X_val, k, sep = ' newline '):
    X_val_split = [x.split( ' newline ') for x in X_val] 
    inds = extractListFromDic(data, 'filtered_line_inds')
    ps = extractListFromDic(data, 'filtered_line_probs')
    for t, x in enumerate(X_val_split):
        ind = inds[t]
        p = ps[t]
        used, start = [], {}
        for i in range(min(len(x), k)):
            for j in range(min(len(x), k)):
                if j!=i:
                    if ind[i] - ind[j] == 1:
                        used.append(i)
                        used.append(j)
                        start[i] = j
                    elif ind[i] - ind[j] == -1:
                        used.append(i)
                        used.append(j)
                        start[j] = i

        if len(start) > 0:
            new_x = []
            new_p = []
            for i in range(min(len(x), k)):
                if i not in used:
                    new_x.append(x[i])
                    new_p.append(p[i])
                else:
                    if i not in start:
                        continue
                    else:
                        new_x.append( x[i] + " " + x[start[i]])
                        new_p.append( max(p[i], p[start[i]]))
        
            X_val_split[t] = new_x   
            data[t]['filtered_line_probs'] = new_p
            data[t]['filtered_lines'] = copy.deepcopy(new_x)
    return X_val_split

def joinAdjacentLinesRules(data, X_val, k):
    X_val_split = [x.split( ' newline ') for x in X_val] 
    inds = extractListFromDic(data, 'filtered_line_inds')
    for t, x in enumerate(X_val_split):
        ind = inds[t]
        used, start = [], {}
        for i in range(len(ind)):
            for j in range(len(ind)):
                if j!=i:
                    if ind[i] - ind[j] == 1:
                        used.append(i)
                        used.append(j)
                        start[i] = j
                    elif ind[i] - ind[j] == -1:
                        used.append(i)
                        used.append(j)
                        start[j] = i

        if len(start) > 0:
            new_x = []
            for i in range(min(len(x), k)):
                if i not in used:
                    new_x.append(x[i])
                else:
                    if i not in start:
                        continue
                    else:
                        new_x.append( x[i] + " " + x[start[i]])
        
            X_val_split[t] = new_x   
    return X_val_split
    
    
def lrchop(thestring, pattern):
    if thestring.endswith(pattern):
        return thestring[:-len(pattern)]
    if thestring.startswith(pattern):
        return thestring[len(pattern):]
    
    return thestring