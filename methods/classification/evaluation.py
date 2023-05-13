import numpy as np

from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.preprocessing import LabelEncoder

def getScores(y_true, y_pred):
    # Encode labels into 0, 1, 2, etc values
    lb = LabelEncoder()
    combined = np.concatenate((y_true, y_pred), axis=0)           
    combined = lb.fit(combined)

    y_true = lb.transform(y_true)
    y_pred = lb.transform(y_pred)
    
    scores = {}
    scores['f1_weighted'] = f1_score(y_true,y_pred, average = 'weighted', pos_label = None )
    scores['f1_macro'] = f1_score(y_true,y_pred, average = 'macro', pos_label = None )
    scores['f1_micro'] = f1_score(y_true,y_pred, average = 'micro', pos_label = None )
    scores['f1_by_class'] = f1_score(y_true,y_pred, average = None, pos_label = None ).tolist()

    scores['precision_weighted'] = precision_score(y_true,y_pred, average = 'weighted', pos_label = None )
    scores['precision_macro'] = precision_score(y_true,y_pred, average = 'macro', pos_label = None )
    scores['precision_micro'] = precision_score(y_true,y_pred, average = 'micro', pos_label = None )
    scores['precision_by_class'] = precision_score(y_true,y_pred, average = None, pos_label = None ).tolist()

    scores['recall_weighted'] = recall_score(y_true,y_pred, average = 'weighted', pos_label = None )
    scores['recall_macro'] = recall_score(y_true,y_pred, average = 'macro', pos_label = None )
    scores['recall_micro'] = recall_score(y_true,y_pred, average = 'micro', pos_label = None )
    scores['recall_by_class'] = recall_score(y_true,y_pred, average = None, pos_label = None ).tolist()
    return scores