import glob
import numpy as np
import os
import random
import re
import sys

def getText(text,perserve_lines=False):
    text_start = "<TEXT><![CDATA[\ufeff"
    n = len(text_start)
    text = " ".join(text)
    ind = text.index(text_start)
    text = text[(ind+n):]
    if not perserve_lines:
        text = text.split("\n")
        text = [x for x in text if x != " "]
        return " NEWLINE ".join(text)
    else:
        return text

def getHighlightedText(xml_output, field_names):
    texts = {field: "NA" for field in field_names}
    for output in xml_output:
        if output[1:4] != "PHI":
            field = extractFieldName(output)
            ind = output.index("text=")
            start_ind = ind + 6
            try:
                end_ind = output.index("type=")
            except:
                end_ind = len(output)-1
    
            text = output[start_ind:end_ind]
            text = text.rstrip(" ").lstrip(" ")
            end_ind = text.index('\"')
            text = text[:(end_ind)].rstrip(" ").lstrip(" ")
            text = " ".join(text.split())
            text = text.replace('&amp;', '&')
            text = text.replace('&gt;', '>')
            text = text.replace('&lt;', '<')
            texts[field] = text
    return texts

def getSpans(xml_output, field_names):
    spans = {field: "NA" for field in field_names}
    for output in xml_output:
        if output[1:4] != "PHI":
            field = extractFieldName(output)
            ind = output.index("spans=")
            start_ind = ind + 7
            end_ind = output.index("text")    
            span = output[start_ind:end_ind]
            span = span.rstrip(" ").lstrip(" ")
            end_ind = span.index('\"')
            span = span[:(end_ind)].rstrip(" ").lstrip(" ")
            span_inds = span.split('~')
            spans[field] = (int(span_inds[0]), int(span_inds[1]))
    return spans

def separate_xml_from_text(file_name):
    with open(file_name,'r') as f:
        lines = f.readlines()
    for i,l in enumerate(lines):
        if l ==  '<TAGS>\n':
            xml_split_line = i
            break
    text_sect = lines[:i-1]
    xml_sect = lines[i+1:-2]
    return text_sect, xml_sect  

def extractFieldName(output):
    field = output[1:(output.index(" "))]
    id_ind = output.index("id")
    field_id = output[(id_ind+4):]
    id_ind = field_id.index("\"")
    field_id = field_id[:id_ind]
    
    return field + "_" + field_id

def getFieldNames(xml_outputs):
    field_names = []
    for xml_output in xml_outputs:
        for output in xml_output:
            if output[1:4] != 'PHI':
                field_name = extractFieldName(output)
                if field_name not in field_names:
                    field_names.append(field_name)
    return field_names

def getNumPossibleFieldValues(labels):
    num_values = {field:0 for field in list(labels[0].keys())}
    field_names = list(labels[0].keys())
    for field in field_names:
        values = []
        for label in labels:
            if label[field] not in values:
                values.append(label[field])
        num_values[field] = len(values)
    return num_values

def getLabelandType(xml_output, field_names):
    labels = {field: "NA" for field in field_names}
    for output in xml_output:
        if output[1:4] != "PHI":
            
            field = extractFieldName(output)
            try:
                ind = output.index("type=")
            except:
                ind = output.index("text=")
            start_ind = ind + 6
            try:
                end_ind = output.index("/>")
            except:
                end_ind = len(output)-1
            label = output[start_ind:end_ind]
            label = label.rstrip(" ").lstrip(" ")
            end_ind = label.index('\"')
            label = label[:(end_ind)].rstrip(" ").lstrip(" ")
            label = " ".join(label.split())
            label = label.replace('&amp;', '&')
            label = label.replace('&gt;', '>')
            label = label.replace('&lt;', '<')
            labels[field] = label.lower()
    return labels

def line_locator(text,spans):
    if spans == 'NA':
        return 'NA'
    index_offset = 0
    start_index = spans[0]
    end_index = spans[1]
    true_index = 0
    for i in range(len(text)):
        if i-1*text[0:i].count('\n') == spans[0]:
            start_index = i-1
        if i-1*text[0:i].count('\n') == spans[1]:
            end_index = i-1            
    j = start_index
    while text[j-1] != '\n':
        j -= 1
    line_start = j
    j = end_index
    while text[j:j+1] != '\n':
        j += 1
    line_end = j
    return ' '.join(text[line_start:line_end].replace('\n',' NEWLINE ').split())

def getLines(xml_output,text,field_names):
    lines = {field: "NA" for field in field_names}
    spans = getSpans(xml_output,field_names)
    for field in field_names:
        raw_line = line_locator(getText(text,True),spans[field])
        line = raw_line.split("\n")
        line = [x for x in line if x != " "]
        line = " NEWLINE ".join(line)
        lines[field] = line
    return lines

def create_label_dict(field_names):    
    label_dict = {}
    for f in field_names:
        if f[:-1] not in label_dict:
            label_dict[f[:-1]] = []
        label_dict[f[:-1]].append(f)
        
    return label_dict

def most_common(lst):
    return max(set(lst), key=lst.count)

def label_compressor(labels,lines,field_names):
    new_labels = []
    new_lines = []
    for i in range(len(labels)):
        doc_labels = labels[i]
        doc_lines = lines[i]
        pre_comp_labels = defaultdict(list)
        pre_comp_lines = defaultdict(list)
        for key in doc_labels.keys():
            label = doc_labels[key]
            doc_line = doc_lines[key]
            if label != 'NA':
                pre_comp_labels[key[:-1]].append(label)
                pre_comp_lines[key[:-1]].append(doc_line)
        comp_labels = dict()
        comp_lines = dict()
        for key in field_names:
            if len(pre_comp_labels[key]) == 0:
                comp_labels[key] = 'NA'
                comp_lines[key] = ['']
            else:
                comp_labels[key] = most_common(pre_comp_labels[key])
                comp_lines[key] = pre_comp_lines[key]
        new_labels.append(comp_labels)
        new_lines.append(comp_lines)
    return new_labels, new_lines, pre_comp_labels,pre_comp_lines

def map_labels(field,old_label,new_label,labels):
    for l in labels:
        for key in l.keys():
            if key[:-1] == field:
                if l[key] == old_label:
                    l[key] = new_label
    return labels

def getUniqueLabels(field,labels):
    f = []
    
    for label in labels:
        f.append(label[field])
    
    return np.unique(f)
    