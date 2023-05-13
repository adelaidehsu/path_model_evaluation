import torch

from methods.interpretations.shap.utils import convert_to_masker
from methods.interpretations.utils import compute_input_type_attention
from methods.interpretations.utils import visualize

class AttentionWeightsWrapper():
    
    def __init__(self, model, model_type, tokenizer, class_names, cuda=True):
        self.model = model
        self.model_type = model_type
        self.tokenizer = tokenizer
        self.cuda = cuda
        self.class_names = class_names
        self.masker = convert_to_masker(self.tokenizer)        
    
    def average_across_heads_single_layer(self, attention_heads_all_layers, i):    
        return torch.mean(attention_heads_all_layers[i].squeeze(0), dim=0)

    def average_across_layers_and_heads(self, attention_heads_all_layers):
        for i, attention_heads in enumerate(attention_heads_all_layers):
            if i == 0:
                attention_weights = torch.mean(attention_heads.squeeze(0), dim=0)
            else:
                attention_weights += torch.mean(attention_heads.squeeze(0), dim=0)
        return attention_weights/len(attention_heads_all_layers)

    def average_across_subset_layers_and_heads(self, attention_heads_all_layers, start_index, end_index):
        for i, j in enumerate(range(start_index, end_index+1)):
            if i == 0:
                attention_weights = torch.mean(attention_heads_all_layers[j].squeeze(0), dim=0)
            else:
                attention_weights += torch.mean(attention_heads_all_layers[j].squeeze(0), dim=0)
        return attention_weights/(end_index + 1 - start_index)

    def interpret(self, text, label, method='all', viz=True, layer_start=None, layer_end=None, layer_ind=None):
        self.model.cpu()
        input_ids, token_type_ids, attention_mask = compute_input_type_attention(text, self.tokenizer)
        all_tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0].detach().tolist())
        outputs = self.model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        attention_heads_all_layers = outputs[-1][-1]

        if method == 'all':
            attention_weights = self.average_across_layers_and_heads(attention_heads_all_layers)
            print(attention_weights.shape)
            summed_attention_weights = attention_weights.sum(0)/439
            print(summed_attention_weights.shape)
            attributions = summed_attention_weights.cpu().detach().numpy()
            print(attributions.shape)
        elif method == 'single':
            assert(layer_ind is not None)
            
            attention_weights = self.average_across_heads_single_layer(attention_heads_all_layers, layer_ind)
            summed_attention_weights = attention_weights.sum(0)/439
            attributions = summed_attention_weights.cpu().detach().numpy()
        else:
            assert(layer_start is not None)
            assert(layer_end is not None)
            
            attention_weights = self.average_across_subset_layers_and_heads(attention_heads_all_layers, indices)
            summed_attention_weights = attention_weights.sum(0)/attention_weights.sum(0).shape[0]
            attributions = summed_attention_weights.cpu().detach().numpy()
        
        attributions = attributions/sum(attributions)
        
        if viz:
            visualize(attributions*20, text, label, self.model, self.tokenizer)
                        
        return attributions, all_tokens