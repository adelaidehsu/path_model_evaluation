import copy 

from collections import defaultdict, Counter

def getNumPossibleFieldValues(data):
    field_names = list(data[0]['labels'].keys())
    num_values = {field:0 for field in field_names}
    
    for field in field_names:
        values = []
        for patient in data:
            if patient['labels'][field] not in values:
                values.append(patient['labels'][field])
        num_values[field] = len(values)
    return num_values

def most_common(lst):
    return max(set(lst), key=lst.count)

def compressLabels(data,field_names,conjoin=True):
    compressed_data = []
    for i, patient in enumerate(data):
        pre_comp_labels, pre_comp_lines, pre_comp_spans = defaultdict(list), defaultdict(list), defaultdict(list)       
        labels = copy.deepcopy(patient['labels'])
        lines = copy.deepcopy(patient['lines'])
        spans = copy.deepcopy(patient['spans'])
        
        for key in labels.keys():
            found = False
            
            for name in field_names:
                if name in key:
                    found = True
            
            if found:
                label = labels[key]
                line = lines[key]
                span = spans[key]
                if label != 'NA':
                    pre_comp_labels[key[:-1]].append(label)
                    pre_comp_lines[key[:-1]].append(line)
                    pre_comp_spans[key[:-1]].append(span)
                
        comp_labels, comp_lines, comp_spans = dict(), dict(), dict()

        for key in field_names:
            if len(pre_comp_labels[key]) == 0:
                comp_labels[key] = 'NA'
                comp_lines[key] = ['']
                comp_spans[key] = ['']
            else:
                if conjoin:
                    unique_labels = list(set(pre_comp_labels[key]))
                    unique_labels.sort()
                    comp_labels[key] = '-'.join(unique_labels)
                else:
                    comp_labels[key] = most_common(pre_comp_labels[key])
                comp_lines[key] = pre_comp_lines[key]
                comp_spans[key] = pre_comp_spans[key]
                
        patient['labels'] = comp_labels
        patient['lines'] = comp_lines
        patient['spans'] = comp_spans
        patient['original_labels'] = labels
        patient['original_lines'] = lines
        patient['original_spans'] = spans
        compressed_data.append(patient)
    return compressed_data

def getNanProportions(data):
    field_names = list(data[0]['labels'].keys())
    num_nans = {field:0 for field in field_names}
    
    for field in field_names:
        nans = 0
        for patient in data:
            labels = patient['labels']
            if labels[field] == 'NA':
                nans+= 1
        num_nans[field] = nans/len(data)
    return num_nans

def findTies(data,field_names):
    ties = []
    
    compressed_data = []
    for i, patient in enumerate(data):
        ret = {key:[] for key in field_names}
        pre_comp_labels, pre_comp_lines, pre_comp_spans = defaultdict(list), defaultdict(list), defaultdict(list)       
        labels = copy.deepcopy(patient['original_labels'])
        lines = copy.deepcopy(patient['original_lines'])
        spans = copy.deepcopy(patient['original_spans'])
        
        for key in labels.keys():
            label = labels[key]
            line = lines[key]
            span = spans[key]
            if label != 'NA':
                pre_comp_labels[key[:-1]].append(label)
                pre_comp_lines[key[:-1]].append(line)
                pre_comp_spans[key[:-1]].append(span)
                
        comp_labels, comp_lines, comp_spans = dict(), dict(), dict()

        for key in field_names:
            if len(pre_comp_labels[key]) == 0:
                comp_labels[key] = 'NA'
                comp_lines[key] = ['']
                comp_spans[key] = ['']
            else:
                dic = Counter(pre_comp_labels[key])
                c = max(dic.values())
                for k in dic:
                    if dic[k] == c:
                        ret[key].append(k)
        ties.append(ret)
    return ties 

def fixLabels(data, domain):
    # Fix annotation errors
    if domain == 'lung':
        data['train'][135]['labels']['Procedure_Pr'] = 'lobectomy'
        data['train'][36]['labels']['TumorSite_T'] = 'lower lobe'
    
    if domain == 'colon':
        data['train'][36]['labels']['HistologicGrade_Hi'] = 'NA'
        data['train'][61]['labels']['HistologicGrade_Hi'] = 'g1-g2'
        data['train'][97]['labels']['HistologicGrade_Hi'] = 'g1'
        data['train'][124]['labels']['HistologicGrade_Hi'] = 'g1'
        data['train'][130]['labels']['HistologicGrade_Hi'] = 'g2'
        data['train'][162]['labels']['HistologicGrade_Hi'] = 'g1'
        data['train'][181]['labels']['HistologicGrade_Hi'] = 'NA'
        data['test'][10]['labels']['HistologicGrade_Hi'] = 'NA'
        data['train'][98]['labels']['HistologicType_H'] = 'adenocarcinoma'
        data['train'][16]['labels']['PerineuralInvasion_Pe'] = 'not identified'
        data['train'][26]['labels']['PerineuralInvasion_Pe'] = 'not identified'
        data['train'][84]['labels']['PerineuralInvasion_Pe'] = 'not identified'
        data['val'][6]['labels']['PerineuralInvasion_Pe'] = 'not identified'
        data['test'][16]['labels']['PerineuralInvasion_Pe'] = 'not identified'
        data['train'][10]['labels']['LymphovascularInvasion_L'] = 'not identified'
        data['train'][32]['labels']['LymphovascularInvasion_L'] = 'present'
        data['train'][89]['labels']['LymphovascularInvasion_L'] = 'present'
        data['val'][10]['labels']['LymphovascularInvasion_L'] = 'present'
        data['test'][10]['labels']['LymphovascularInvasion_L'] = 'present'
        data['train'][49]['labels']['Procedure_Pr'] = 'transverse colectomy'
        data['train'][64]['labels']['Procedure_Pr'] = 'left hemicolectomy'
        data['train'][52]['labels']['Procedure_Pr'] = 'sigmoidectomy'
        data['train'][65]['labels']['Procedure_Pr'] = 'sigmoidectomy'
        data['train'][81]['labels']['Procedure_Pr'] = 'right hemicolectomy'
        data['train'][168]['labels']['Procedure_Pr'] = 'left hemicolectomy'
        data['val'][7]['labels']['Procedure_Pr'] = 'right hemicolectomy'
        data['val'][26]['labels']['Procedure_Pr'] = 'sigmoidectomy'
        data['val'][28]['labels']['Procedure_Pr'] = 'sigmoidectomy'
        data['test'][0]['labels']['Procedure_Pr'] = 'sigmoidectomy'
        data['train'][87]['labels']['TumorSite_T'] = 'cecum'
        data['train'][125]['labels']['TumorSite_T'] = 'cecum'
        data['train'][137]['labels']['TumorSite_T'] = 'cecum'
        data['train'][141]['labels']['TumorSite_T'] = 'cecum'
        data['train'][144]['labels']['TumorSite_T'] = 'hepatic flexure'
        data['train'][158]['labels']['TumorSite_T'] = 'cecum'
        data['val'][0]['labels']['TumorSite_T'] = 'cecum'
        data['val'][10]['labels']['TumorSite_T'] = 'rectosigmoid junction'
        data['val'][35]['labels']['TumorSite_T'] = 'transverse colon'
        data['test'][11]['labels']['TumorSite_T'] = 'rectum'
        data['test'][18]['labels']['TumorSite_T'] = 'cecum'
        data['test'][21]['labels']['TumorSite_T'] = 'cecum'
        data['test'][23]['labels']['TumorSite_T'] = 'transverse colon'
    
    if domain == 'kidney':
        # Fix annotation labels
        data['train'][43]['labels']['HistologicGrade_Hi'] = 'g1-g2'
        data['train'][66]['labels']['HistologicGrade_Hi'] = 'g1'
        data['train'][67]['labels']['HistologicGrade_Hi'] = 'g3'
        data['train'][126]['labels']['HistologicGrade_Hi'] = 'g2-g3'
        data['train'][148]['labels']['HistologicGrade_Hi'] = 'g2'
        data['train'][152]['labels']['HistologicGrade_Hi'] = 'g3'
        data['train'][153]['labels']['HistologicGrade_Hi'] = 'g3'
        data['train'][166]['labels']['HistologicGrade_Hi'] = 'g3'
        data['train'][170]['labels']['HistologicGrade_Hi'] = 'g2-g3'
        data['train'][179]['labels']['HistologicGrade_Hi'] = 'g3-g4'
        data['val'][9]['labels']['HistologicGrade_Hi'] = 'g1'
        data['test'][14]['labels']['HistologicGrade_Hi'] = 'g1-g2'
        data['test'][15]['labels']['HistologicGrade_Hi'] = 'g1'
        data['train'][22]['labels']['LymphovascularInvasion_L'] = 'not identified'
        data['train'][134]['labels']['LymphovascularInvasion_L'] = 'not identified'
        data['test'][19]['labels']['LymphovascularInvasion_L'] = 'not identified'
        data['test'][25]['labels']['LymphovascularInvasion_L'] = 'not identified'
        data['train'][16]['labels']['SpecimenLaterality_S'] = 'right'
        data['train'][80]['labels']['SpecimenLaterality_S'] = 'right'
        data['val'][19]['labels']['SpecimenLaterality_S'] = 'left'
        data['val'][20]['labels']['SpecimenLaterality_S'] = 'left'
        data['train'][1]['labels']['TumorSite_T'] = 'lower-mid-upper'
        data['train'][4]['labels']['TumorSite_T'] = 'mid-upper'
        data['train'][18]['labels']['TumorSite_T'] = 'mid-lower'
        data['train'][21]['labels']['TumorSite_T'] = 'mid'
        data['train'][25]['labels']['TumorSite_T'] = 'mid'
        data['train'][32]['labels']['TumorSite_T'] = 'upper'
        data['train'][34]['labels']['TumorSite_T'] = 'lower-mid'
        data['train'][62]['labels']['TumorSite_T'] = 'mid'
        data['train'][63]['labels']['TumorSite_T'] = 'lower-upper'
        data['train'][64]['labels']['TumorSite_T'] = 'lower-upper'
        data['train'][68]['labels']['TumorSite_T'] = 'mid-upper'
        data['train'][83]['labels']['TumorSite_T'] = 'mid-upper'
        data['train'][86]['labels']['TumorSite_T'] = 'lower-upper'
        data['train'][93]['labels']['TumorSite_T'] = 'lower-mid'
        data['train'][124]['labels']['TumorSite_T'] = 'mid-upper'
        data['train'][127]['labels']['TumorSite_T'] = 'mid-upper'
        data['train'][134]['labels']['TumorSite_T'] = 'other'
        data['train'][142]['labels']['TumorSite_T'] = 'lower-mid'
        data['train'][167]['labels']['TumorSite_T'] = 'mid-upper' # OKAY
        data['train'][168]['labels']['TumorSite_T'] = 'lower-mid-upper'
        data['train'][170]['labels']['TumorSite_T'] = 'mid-upper'
        data['train'][177]['labels']['TumorSite_T'] = 'lower-other'
        data['train'][179]['labels']['TumorSite_T'] = 'other'
        data['val'][12]['labels']['TumorSite_T'] = 'upper'
        data['val'][16]['labels']['TumorSite_T'] = 'lower'
        data['val'][21]['labels']['TumorSite_T'] = 'lower-upper'
        data['val'][22]['labels']['TumorSite_T'] = 'mid-upper'
        data['val'][35]['labels']['TumorSite_T'] = 'other'
        data['test'][10]['labels']['TumorSite_T'] = 'lower-mid'
        data['test'][16]['labels']['TumorSite_T'] = 'mid-upper'
        data['test'][19]['labels']['TumorSite_T'] = 'lower-mid-upper'
        data['train'][13]['labels']['HistologicType_H'] = 'papillary renal cell carcinoma type 1' #OK
        data['train'][30]['labels']['HistologicType_H'] = 'other histologic type'
        data['train'][69]['labels']['HistologicType_H'] = 'papillary renal cell carcinoma type 2' #OK
        data['train'][106]['labels']['HistologicType_H'] = 'papillary renal cell carcinoma type 2' #OK
        data['train'][144]['labels']['HistologicType_H'] = 'renal cell carcinoma unclassified' 
        data['train'][150]['labels']['HistologicType_H'] = 'papillary renal cell carcinoma type 2' #OK
        data['train'][158]['labels']['HistologicType_H'] = 'clear cell renal cell carcinoma' 
        data['train'][164]['labels']['HistologicType_H'] = 'papillary renal cell carcinoma type 1' #OK
        data['train'][165]['labels']['HistologicType_H'] = 'clear cell renal cell carcinoma' 
        data['train'][182]['labels']['HistologicType_H'] = 'clear cell renal cell carcinoma' 
        data['train'][183]['labels']['HistologicType_H'] = 'papillary renal cell carcinoma type 2' #OK
        data['val'][0]['labels']['HistologicType_H'] = 'papillary renal cell carcinoma type 2' #OK
        data['val'][2]['labels']['HistologicType_H'] = 'clear cell renal cell carcinoma' 
        data['val'][12]['labels']['HistologicType_H'] = 'papillary renal cell carcinoma type 1' #OK
        data['val'][16]['labels']['HistologicType_H'] = 'papillary renal cell carcinoma type 1' #OK
        data['val'][17]['labels']['HistologicType_H'] = 'clear cell renal cell carcinoma' 
        data['val'][30]['labels']['HistologicType_H'] = 'clear cell renal cell carcinoma' 
        data['test'][7]['labels']['HistologicType_H'] = 'papillary renal cell carcinoma type 2' #OK
        data['test'][15]['labels']['HistologicType_H'] = 'mucinous tubular and spindle renal cell carcinoma' 
        data['test'][16]['labels']['HistologicType_H'] = 'papillary renal cell carcinoma type 2' #OK
        data['test'][23]['labels']['HistologicType_H'] = 'papillary renal cell carcinoma type 2' #OK
        data['test'][25]['labels']['HistologicType_H'] = 'papillary renal cell carcinoma type 1' #OK
        
        data['train'][1]['labels']['Procedure_Pr'] = 'radical nephrectomy'
        data['train'][2]['labels']['Procedure_Pr'] = 'radical nephrectomy'
        data['train'][61]['labels']['Procedure_Pr'] = 'radical nephrectomy'
        data['train'][35]['labels']['Procedure_Pr'] = 'radical nephrectomy'
        data['train'][138]['labels']['Procedure_Pr'] = 'radical nephrectomy'
        data['train'][157]['labels']['Procedure_Pr'] = 'radical nephrectomy'
        data['train'][178]['labels']['Procedure_Pr'] = 'radical nephrectomy'
        data['val'][2]['labels']['Procedure_Pr'] = 'radical nephrectomy'
        data['test'][1]['labels']['Procedure_Pr'] = 'radical nephrectomy'
        data['test'][25]['labels']['Procedure_Pr'] = 'radical nephrectomy'
        
    return data