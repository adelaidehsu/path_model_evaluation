# -*- coding: utf-8 -*-
import json
import numpy as np
import os

from pyfunctions.general import hasLetter, hasNumeric

integerMapping = {'one': 1, 'two':2, 'three': 3, 'four':4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine':9,
          'ten': 10, 'eleven': 11, 'twelve':12, 'thirteen': 13,'fourteen': 14, 'fifteen': 15, 'sixteen': 16,
          'seventeen': 17, 'eighten': 18, 'nineteen': 19, 'twenty': 20}

def extractAccessionId(texts):
    processed, flip_NA = [], []
    
    for text in texts:
        text = text.lower().split()
        here = False
        for word in text:
            if hasNumeric(word) and hasLetter(word) and '-' in word and not here:
                processed.append(word)
                here = True
                flip_NA.append(0)
        if not here:
            processed.append('<na>')
            flip_NA.append(1)
    return processed, flip_NA
    
def extractMrn(texts):
    processed, flip_NA = [], []
    for text in texts:
        text = text.lower().split()
        here = False
        for word in text:
            if hasNumeric(word) and len(word) > 4 and not here:
                processed.append(word)
                here = True
                flip_NA.append(0)
        if not here:
            processed.append('<na>')
            flip_NA.append(1)
    return processed, flip_NA

def extractTumorSize(texts):
    processed, flip_NA = [], []
    for i, text in enumerate(texts):
        text = text.lower().split()
        here = False

        candidates = []
        for word in text:
            if hasNumeric(word) and not hasLetter(word):
                try:
                    candidates.append(float(word))
                    here = True
                except:
                    continue
        if not here:
            processed.append('<na>')
            flip_NA.append(1)
        else:
            processed.append(str(max(candidates)))
            flip_NA.append(0)
    return processed, flip_NA

def extractExaminedNodes(texts):
    processed, flip_NA = [], []
    for i, text in enumerate(texts):
        text = text.replace("(", "")
        text = text.replace(")", "")
        text = text.lower().split()
        flag = False

        candidates = []
        for word in text:
            if hasNumeric(word) and not hasLetter(word):
                try:
                    candidates.append(float(word))
                    flag = True
                except:
                    continue
            if word in integerMapping:
                try:
                    candidates.append(float(integerMapping[word]))
                    flag = True
                except:
                    continue
        if not flag:
            processed.append('<na>')
            flip_NA.append(1)
        else:
            processed.append(str(max(candidates)))
            flip_NA.append(0)
    return processed, flip_NA

def extractInvolvedNodes(texts):
    processed, flip_NA = [], []
    for i, text in enumerate(texts):
        text = text.replace("(", "")
        text = text.replace(")", "")
        text = text.replace("/", " / ")
        text = text.lower().split()
        flag1, flag2 = False, False

        candidates = []
        for word in text:
            if hasNumeric(word) and not hasLetter(word):
                try:
                    candidates.append(float(word))
                    flag1 = True
                except:
                    continue
            if word in integerMapping:
                try:
                    candidates.append(float(integerMapping[word]))
                    flag1 = True
                except:
                    continue
        for word in text:
            if 'negative' in word and not flag1 and not flag2 or 'no' == word and not flag1 and not flag2:
                processed.append('negative')
                flip_NA.append(0)
                flag2 = True
            elif 'positive' in word and not flag1 and not flag2:
                processed.append('positive')
                flip_NA.append(0)
                flag2 = True
        if not flag1 and not flag2:
            processed.append('<na>')
            flip_NA.append(1)
        else:
            if not flag2:
                processed.append(str(min(candidates)))
                flip_NA.append(0)
    return processed, flip_NA

def extractStage(texts):
    processed, flip_NA = [], []
    for i, text in enumerate(texts):
        text = str(text)
        text = text.replace("/", " / ")
        text = text.replace("(", " ")
        text = text.replace(")", " ")
        text = text.lower()
        flag = False
        for word in text.split():
            if (word[:2] == 'pt' or word[:2] == 'yp') and not flag:
                processed.append(word)
                flip_NA.append(0)
                flag = True
        if not flag:
            processed.append("<na>")
            flip_NA.append(1)
    return processed, flip_NA

def extractLabel(texts, field, ret_flipNA = False):
    field = field.lower()
    if 'accession' in field:
        labels = extractAccessionId(texts)
    elif 'mrn' in field:
        labels = extractMrn(texts)
    elif 'tumorsize' in field:
        labels = extractTumorSize(texts)
    elif 'lymphnodesexamined' in field:
        labels = extractExaminedNodes(texts)
    elif 'lymphnodesinvolved' in field:
        labels = extractInvolvedNodes(texts)
    elif '_p' in field:
        labels= extractStage(texts)
    
    if ret_flipNA:
        return labels
    else:
        return labels[0]