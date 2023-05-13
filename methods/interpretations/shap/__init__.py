import numpy as np
import scipy as sp
import shap
import torch
import torch.nn.functional as F

from methods.interpretations.shap.utils import convert_to_masker
from methods.interpretations.utils import visualize
from shap import maskers

class ShapWrapper():
    
    def __init__(self, model, model_type, tokenizer, class_names, cuda=True):
        self.model = model
        self.model_type = model_type
        self.tokenizer = tokenizer
        self.cuda = cuda
        self.class_names = class_names
        self.masker = convert_to_masker(self.tokenizer)
 
    def forward(self, x):
        tv = torch.tensor([self.tokenizer.encode(v, padding='max_length', max_length=500, truncation=True) for v in x]).cuda()
        outputs = self.model(tv)[0].detach().cpu().numpy()
        scores = (np.exp(outputs).T / np.exp(outputs).sum(-1)).T
        val = sp.special.logit(scores)
        return val

    def interpret(self, text, label, viz=True):
        if self.cuda:
            self.model.cuda()
        else:
            self.model.cpu()
            
        inputs = {'label': np.array([label]), 'text': np.array([text])}
        
        explainer = shap.explainers.Partition(self.forward, self.masker)
        shap_values = explainer(inputs)
        attributions = shap_values[:,:,label].values[0]
        if viz:
            #shap.plots.text(shap_values[:,:,label])
            visualize(attributions, text, label, self.model, self.tokenizer)
        
        return attributions,  shap_values.data[0].tolist()