import copy
import sys

import numpy as np
import pandas as pd

from collections import Counter, defaultdict
from scipy.sparse import csr_matrix, hstack, vstack


def checkTies(data, predictions, ties, field):
    # Override label with prediction if prediction is one of the classes
    # that are tied for the most amount of appearances in annotation
    # ties contains all the predictions with most amount of appearances
    for i in range(len(data)):
        pred = predictions['predicted_label'].iloc[i]
        
        if pred in ties[i][field]:
            predictions['label'].iloc[i] = pred
    return predictions

def getPredictions(data, X_split, NA, field, classifier, args , probs = False):
    if args['method'] == 'weight_labels' or args['method'] == 'weighted_sum':
        return getPredictionsWeighted(data, X_split, NA, field, classifier, args)
    elif args['method'] == 'concatenate':
        return getPredictionsConcatenated(data, X_split, NA, field, classifier, args)
    elif args['method'] == 'oracle':
        return getPredictionsConcatenated(data, X_split, NA, field, classifier, args)
    elif args['method'] == 'rules':
        if args['rules'] == 'concatenate':
            return getPredictionsConcatenated(data, X_split, NA, field, classifier, args)
        else:
            return getPredictionsRulesWeighted(data, X_split, NA, field, classifier, args)
    elif args['method'] == 'string_sim':
        return getPredictionsStringSim(data, X_split, NA, field, classifier, args, hypers)
    
        
    return getPredictionsConcatenated(data, X_split, NA, field, classifier, args, probs)
        
def processPredictions(pred, data, k, args):
    if args['method'] == 'weight_labels':
        return processPredictionsWeighted(pred, data, k, args)
    elif args['method'] == 'weighted_sum':
        return processPredictionsWeightedSum(pred, data, k, args)
    elif args['method'] == 'concatenate':
        return processPredictionsConcatenated(pred, data, k, args)
    elif args['method'] == 'oracle':
        return processPredictionsConcatenated(pred, data, k, args)
    elif args['method'] == 'rules':
        if args['rules'] == 'concatenate':
            return processPredictionsConcatenated(pred, data, k, args)
        elif args['rules'] == 'weighted_sum':
            return processPredictionsRulesWeightedSum(pred, data, k, args)
        elif args['rules'] == 'weights':
            return processPredictionsRulesWeighted(pred, data, k, args)
    elif args['method'] == 'string_sim':
        return processPredictionsStringSim(data, X_split, NA, field, classifier, args)
            
    return processPredictionsConcatenated(pred, data, k, args)

def getPredictionsConcatenated(data, X_split, NA, field, classifier, args, proba = False):
    predictions = pd.DataFrame(columns = ['id', 'lines', 'predicted_lines', 'label', 'predicted_label', 'y_prob'])
    pred = classifier.predict(vstack(X_split))
    probs = classifier.predict_proba(vstack(X_split))
    for i, patient in enumerate(data):
        prob = " ".join(probs[i].astype(str))
        predictions.loc[i] = [patient['name'], " ".join(patient['lines'][field]), patient['filtered_lines'] , patient['labels'][field], pred[i], prob]
        
        
    if proba:
        return predictions, probs
    return predictions

def getPredictionsRulesWeighted(data, X_split, NA, field, classifier, args):
    predictions = pd.DataFrame(columns = ['id', 'lines', 'predicted_lines', 'label', 'predicted_label', 'y_prob'])
    
    for i, patient in enumerate(data):
        X_val = X_split[i]
        pred_val = classifier.predict(X_val)
        probs_val = classifier.predict_proba(X_val)
        pred = "|".join(pred_val)

        probs = ""
        for j in range(len(probs_val)):
            probs+= " ".join(probs_val[j].astype(str)) + '|'
        probs = probs[:-1] # get rid of last |

        if NA[i] == 0:
            predictions.loc[i] = [patient['name'], " ".join(patient['lines'][field]), patient['filtered_lines'], patient['labels'][field], pred, probs]
        else:
            NA_overide = "|".join(['NA' for s in patient['filtered_line_inds']])
            predictions.loc[i] = [patient['name'], " ".join(patient['lines'][field]), patient['filtered_lines'], patient['labels'][field], NA_overide, probs]
    return predictions


def getPredictionsStringSim(data, X_split, NA, field, classifier, args, hypers):
    
    # Initialize dataframe to contains predictions and probabilities
    predictions = pd.DataFrame(columns = ['id', 'lines', 'predicted_lines', 'label', 'predicted_label', 'line_prob','line_ind', 
                                          'y_prob'])
    label_set = copy.deepcopy(args['classes'])
    label_set.remove('NA')
    
    weighted_sims = Counter()
    for i, patient in enumerate(data):
        filtered_lines = patient['filtered_lines'].split(args['sep'])
        filtered_line_probs = "|".join([str(s) for s in patient['filtered_line_probs']]) 
        for j, line in enumerate(filtered_lines):
            sims = getSims(text, label_set)
            for l in label_set:
                weighted_sims += filtered_line_probs[j]*sims[l]
        best_label, best_score = weighted_sims.most_common()
        if best_score < hypers['string_sim_thresh']:
            best_label = 'NA'
        
        # Get the line probabilities and indices as string separated by |
        p = "|".join([str(s) for s in patient['filtered_line_probs']])
        q = "|".join([str(s) for s in patient['filtered_line_inds']])
        pred = best_label
        probs = []
        for label in args['classes']:
            if label == 'NA':
                probs.append(1-best_score)
            else:
                probs.append(weighted_sims[label])
        probs = " ".join(probs)
        predictions.loc[i] = [patient['name'], " ".join(patient['lines'][field]), patient['filtered_lines'], patient['labels'][field], pred, p, q, probs]
        
        
    return predictions
                       
        
def processPredictionsStringSim(pred, data, k, args):       
    pred['label'] = pred['label'].astype(str)
    pred['label'].loc[pred['label'] == 'nan'] = 'NA'
    pred['predicted_label'] = pred['pred'].astype(str)
    return pred
            

    
def getPredictionsWeighted(data, X_split, NA, field, classifier, args):
    # This works for both the weighted_labels and weighted_sums method
    # data is the general data for a given split among train, val, test 
    # X_split is a list of sparse matrices where each sparse matrix represents a patient
    ### Each row in the sparse matrix representes a predicted line
    # NA contains an array of zeros and ones, where a 1 means override the prediction for a
    ### report with a 'NA' value
    # field is the field we are interested in
    # classifier is the label classifier for the field
    # args is a dictionary of arugments
    
    # Initialize dataframe to contains predictions and probabilities
    predictions = pd.DataFrame(columns = ['id', 'lines', 'predicted_lines', 'label', 'predicted_label', 'line_prob','line_ind', 
                                          'y_prob'])
    
    for i, patient in enumerate(data):
        # Get sparse matrix for patient i
        X_val = X_split[i]
        
        # Get prediction and label probabilities for each line for a patient
        pred_val = classifier.predict(X_val)
        probs_val = classifier.predict_proba(X_val)
        
        # Save predictions and probabilties for each line as string separated by |
        pred = "|".join(pred_val)
        probs = ""
        
        for j in range(len(probs_val)):
            probs+= " ".join(probs_val[j].astype(str)) + '|'
        probs = probs[:-1] # get rid of last |
        
        # Get the line probabilities and indices as string separated by |
        p = "|".join([str(s) for s in patient['filtered_line_probs']])
        q = "|".join([str(s) for s in patient['filtered_line_inds']])
        
        # Override prediction with NA if NA[i] is a 1
        if NA[i] == 0:
            predictions.loc[i] = [patient['name'], " ".join(patient['lines'][field]), patient['filtered_lines'], patient['labels'][field], pred, p, q, probs]
        else:
            NA_overide = "|".join(['NA' for s in patient['filtered_line_inds']])
            predictions.loc[i] = [patient['name'], " ".join(patient['lines'][field]), patient['filtered_lines'], patient['labels'][field], NA_overide, p, q, probs]
            
    return predictions

def processPredictionsRulesWeighted(pred, data, k, args):
    pred['individual_predictions'] = pred['predicted_label']
    labels = pred['individual_predictions']
    y_probs = pred['y_prob']
    pred['predicted_label'] = 0
    
    for i, label in enumerate(labels):
        full_NA = 'NA'
        all_NAs = [full_NA]
        for i in range(len(label.split("|"))-1):
            full_NA += '|NA'
            all_NAs.append(full_NA)
        
        if label in all_NAs:
            pred['predicted_label'].iloc[i] = 'NA'
        elif str(label).lower() == 'nan':
            pred['predicted_label'].iloc[i] = 'NA'
        else:
            probs = y_probs.iloc[i]
            dic = defaultdict(int)
            
            probs = probs.split("|")
            for j, prob in enumerate(probs):
                if label.split('|')[j] != 'NA':
                    weight = np.max(np.array(prob.split()).astype(float))
                    dic[label.split('|')[j]]+= weight
            m = 0    
            final = 0
            for key in dic:
                if dic[key] > m:
                    m = dic[key]
                    final = key
            pred['predicted_label'].iloc[i] = final
    
    pred['label'] = pred['label'].astype(str)
    pred['label'].loc[pred['label'] == 'nan'] = 'NA'
    pred['predicted_label'] = pred['predicted_label'].astype(str)
    return pred

def processPredictionsRulesWeightedSum(pred, data, k, args):
    pred['individual_predictions'] = pred['predicted_label']
    labels = pred['individual_predictions']
    y_probs = pred['y_prob']
    
    pred['predicted_label'] = 0
        
    for i, label in enumerate(labels):
        if str(label).lower() == 'nan':
            pred['predicted_label'].iloc[i] = 'NA'
        else:
            probs = y_probs.iloc[i]            
            probs = probs.split("|")
            
            sums = np.zeros(len(probs[0].split()))
            for j, prob in enumerate(probs):
                prob = prob.split()
                sums += np.array(prob).astype(float)
            
            pred['predicted_label'].iloc[i] = args['classes'][np.argmax(sums)]
    
    pred['label'] = pred['label'].astype(str)
    pred['label'].loc[pred['label'] == 'nan'] = 'NA'
    pred['predicted_label'] = pred['predicted_label'].astype(str)
    return pred

def processPredictionsWeightedSum(pred, data, k, args):
    # This will take in line inputs and get final predictions for the data
    # data is the main data structure for containing labels, lines, report and etc. for a given split
    # k is the top k lines
    # args is the dictionary of parameters
    # pred is a dataframe
    ### lines column contains the concatenated oracle lines as a string 
    ### predicted_lines column contains the top k lines contatenated by 'newline'
    ### label column contains the label as string like 'g3'
    ### predicted_label column contains the predicted labels for each predicted line as string like 'g1|g2|g1' if k=3
    ### line_prob column contains the line probability for each predicted line as string like '0.99|0.98|0.05' if k=3
    ### y_prob column contains the probabilities for each class for each line separated by |
    ####### for example if its binary and k=3: '0.5 0.5| 0.6 0.4| 0.7 0.3|'
    ### line_ind column contains the indices of each predicted line like '85|56|1|' if k=3
    ### Note if adjacent lines are joined, the number of predicted lines may be less than k
    
    pred['individual_predictions'] = pred['predicted_label']
    labels = pred['individual_predictions']
    y_probs = pred['y_prob']
    
    # Contains the final prediction for the report
    pred['predicted_label'] = 0
    
    for i, label in enumerate(labels):
        if str(label).lower() == 'nan':
            # Check for nan, unsure if obsolete
            pred['predicted_label'].iloc[i] = 'NA'
        else:
            probs = y_probs.iloc[i]
            # Get label probabilities as a list for each line
            ### for example if binary and k=3: ["0.5 0.5", "0.7 0.3", '0.6 0.4']
            probs = probs.split("|")
            
            # Array for containing summed probabilities for each class across lines
            ### Index 0 contains the label probability for the first class, and so on 
            sums = np.zeros(len(probs[0].split()))
            
            for j, prob in enumerate(probs):
                # Split class probabilities into list "0.5 0.5" -> ['0.5', '0.5']
                prob = prob.split()
                if j < len(label.split('|')) and label.split('|')[j] == 'NA' and args['allow_na'] == False:
                    # DO NOTHING. This will keep the probability of NA at zero
                    0
                elif args['weighted_sum'] == 'weighted':
                    # Sum by weighting label probabilities by line prediction
                    if j < len(pred['line_prob'].iloc[i].split("|")):
                        sums += np.array(prob).astype(float)*float(pred['line_prob'].iloc[i].split("|")[j])
                elif args['weighted_sum'] == 'unweighted':
                    # Take sum without weights
                    sums += np.array(prob).astype(float)
                else:
                    # Wrong argument for weighted_sum
                    print('Error in argument')
            
            # Take the class with the max label probability after summing as the final prediction
            pred['predicted_label'].iloc[i] = args['classes'][np.argmax(sums)]
    
    # Make sure labels and predicted labels are strings and preprocess 'nans' to 'NA'
    pred['label'] = pred['label'].astype(str)
    pred['label'].loc[pred['label'] == 'nan'] = 'NA'
    pred['predicted_label'] = pred['predicted_label'].astype(str)
    return pred

def processPredictionsWeighted(pred, data, k, args):
    # This will take in line inputs and get final predictions for the data
    # data is the main data structure for containing labels, lines, report and etc. for a given split
    # k is the top k lines
    # args is the dictionary of parameters
    # pred is a dataframe
    ### lines column contains the concatenated oracle lines as a string 
    ### predicted_lines column contains the top k lines contatenated by 'newline'
    ### label column contains the label as string like 'g3'
    ### predicted_label column contains the predicted labels for each predicted line as string like 'g1|g2|g1' if k=3
    ### line_prob column contains the line probability for each predicted line as string like '0.99|0.98|0.05' if k=3
    ### y_prob column contains the probabilities for each class for each line separated by |
    ####### for example if its binary and k=3: '0.5 0.5| 0.6 0.4| 0.7 0.3|'
    ### line_ind column contains the indices of each predicted line like '85|56|1|' if k=3
    ### Note if adjacent lines are joined, the number of predicted lines may be less than k
    
    pred['individual_predictions'] = pred['predicted_label']
    labels = pred['individual_predictions']
    y_probs = pred['y_prob']
    
    # Contains final prediction for a report
    pred['predicted_label'] = 0
    
    # Make list of NAs, ['NA', 'NA|NA', 'NA|NA|NA'] to check later if all line predictions are NAs
    full_NA = 'NA'
    all_NAs = [full_NA]
    for i in range(k-1):
        full_NA += '|NA'
        all_NAs.append(full_NA)
    
    for i, label in enumerate(labels):
        if label in all_NAs:
            # Check if all lines are NAs then predict NA
            if args['allow_na']:
                pred['predicted_label'].iloc[i] = 'NA'
        elif str(label).lower() == 'nan':
            # Check for nan and predict NA
            pred['predicted_label'].iloc[i] = 'NA'
        else:
            probs = y_probs.iloc[i]
            
            # Dictionary to hold weights for each class
            dic = defaultdict(int)
            
            # Get probs as a list for example if binary and k=3, ["0.5 0.5", '0.7 0.3', '0.8 0.2']
            probs = probs.split("|")
            for j, prob in enumerate(probs):
                
                # Ignore weight if the label is 'NA'
                if j < len(label.split('|')) and j < len(pred['line_prob'].iloc[i].split("|")) and label.split('|')[j] != 'NA':
                    
                    # Take weight as line probability, label probability, or the product of both
                    if args['weights'] == 'product':
                        weight = np.max(np.array(prob.split()).astype(float))*float(pred['line_prob'].iloc[i].split("|")[j])
                    elif args['weights'] == 'line':
                        weight = float(pred['line_prob'].iloc[i].split("|")[j])
                    elif args['weights'] == 'label':
                        weight = np.max(np.array(prob.split()).astype(float))
                    else:
                        print("error")
                    
                    # Add to class weight
                    dic[label.split('|')[j]]+= weight
                    
            # Search class with highest weight
            m = 0    
            final = 0
            for key in dic:
                if dic[key] > m:
                    m = dic[key]
                    final = key
            # Use class as prediction
            pred['predicted_label'].iloc[i] = final
    
    # Process label and predictio nas strings and check for nan
    pred['label'] = pred['label'].astype(str)
    pred['label'].loc[pred['label'] == 'nan'] = 'NA'
    pred['predicted_label'] = pred['predicted_label'].astype(str)
    return pred

def processPredictionsConcatenated(pred, data, k, args):
    pred['label'] = pred['label'].astype(str)
    pred['predicted_label'] = pred['predicted_label'].astype(str)
    pred['label'].loc[pred['label'] == 'nan'] = 'NA'
    return pred
    
    