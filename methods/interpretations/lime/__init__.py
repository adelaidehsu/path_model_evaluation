import numpy as np
import torch
import torch.nn.functional as F

from lime.lime_text import LimeTextExplainer
from methods.interpretations.utils import visualize

class LimeWrapper():
    
    def __init__(self, model, tokenizer, class_names, num_samples=2000, distance_metric='cosine', cuda=True):
        self.model = model
        self.cuda = cuda
        self.tokenizer = tokenizer
        self.class_names = class_names
        self.interpreter = LimeTextExplainer(class_names=class_names)
        self.num_samples = num_samples
        self.distance_metric = distance_metric
        
    def interpret(self, text, label, viz=True):
        if self.cuda:
            self.model.cuda()
        else:
            self.model.cpu()
        
        tokens = self.tokenizer.encode(text)
        tokens = self.tokenizer.convert_ids_to_tokens(tokens)
        label=int(label)
        
        exp = self.interpreter.explain_instance(text, self.forward, labels=(label,), num_features=len(tokens), 
                                 num_samples=self.num_samples, distance_metric=self.distance_metric)
            
        attributions = exp.local_exp[label]
        output_attributions = np.zeros(len(tokens))  
        
        # Get weights for each token
        atts = exp.domain_mapper.map_exp_ids(exp.local_exp[label])
        atts_tokens = [att[0] for att in atts]
        weights = [att[1] for att in atts]

        for i in range(len(atts_tokens)):
            inds = np.where(atts_tokens[i] == np.array(tokens))[0]

            if len(inds) > 0:
                for ind in inds:
                    output_attributions[ind] = weights[i]
        
        if viz:
            visualize(output_attributions, text, label, self.model, self.tokenizer)
        
        return output_attributions, tokens
        
    def forward(self, texts):
        probas = []
        n = len(texts)
        BATCH_SIZE = 4
        num_batches = n//BATCH_SIZE

        probas = []

        for i in range(num_batches):
            encoding = self.tokenizer.batch_encode_plus(texts[i*BATCH_SIZE:(i+1)*BATCH_SIZE], 
                                             add_special_tokens=True, 
                                             max_length=512,
                                             truncation=True, 
                                             padding = "max_length", 
                                             return_attention_mask=True, 
                                             pad_to_max_length=True,
                                             return_tensors="pt")
            
            encoding = encoding.to('cuda')
            outputs = self.model(**encoding)
            proba = F.softmax(outputs[0]).detach().cuda().cpu().numpy()

            probas.append(proba)

        return np.concatenate(probas)