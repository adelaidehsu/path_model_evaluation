import torch
import torch.nn.functional as F

from turing.src.tnlr.modeling import relative_position_bucket

from methods.interpretations.utils import compute_input_type_attention
from methods.interpretations.utils import visualize

class SaliencyMapWrapper():
    
    def __init__(self, model, model_type, tokenizer, class_names):
        self.model = model
        self.model_type = model_type
        self.tokenizer = tokenizer
        self.class_names = class_names

    def interpret(self, text, label, viz=True):
        
        self.model.cpu()
        
        input_ids, token_type_ids, attention_mask = compute_input_type_attention(text, self.tokenizer)
        all_tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0].detach().tolist())
        
        if self.model_type == 'tnlr':
            attributions = compute_tnlr_saliency_attributions(self.model, input_ids, token_type_ids, attention_mask)
        else:    
            attributions = compute_bert_saliency_attributions(text, self.model, self.tokenizer)
              
        if viz:
            attributions = attributions * 4
            visualize(attributions, text, label, self.model, self.tokenizer)
            
        return attributions, all_tokens

def _calculate_rel_pos(model, embedding_output, position_ids):
    if model.bert.config.rel_pos_bins > 0:
        rel_pos_mat = position_ids.unsqueeze(-2) - position_ids.unsqueeze(-1)
        rel_pos = relative_position_bucket(
            rel_pos_mat,
            num_buckets=model.bert.config.rel_pos_bins,
            max_distance=model.bert.config.max_rel_pos,
        )
        rel_pos = F.one_hot(rel_pos, num_classes=model.bert.config.rel_pos_bins).type_as(
            embedding_output
        )
        rel_pos = model.bert.rel_pos_bias(rel_pos).permute(0, 3, 1, 2)
    else:
        rel_pos = None
        
    return rel_pos

def _compute_attributions(scores, grad_output):
    score_max_index = scores.argmax()
    score_max = scores[0,score_max_index]
    score_max.backward()
    saliency, _ = torch.max(grad_output.grad.data,dim=0)

    saliency_token = saliency.sum(dim=1)
    values = saliency_token.abs()/saliency_token.abs().sum()
    return values.numpy()

def compute_tnlr_saliency_attributions(model, input_ids, token_type_ids, attention_mask):

    input_shape = input_ids.size()
    device = input_ids.device

    extended_attention_mask = attention_mask[:, None, None, :]

    extended_attention_mask = extended_attention_mask.to(
        dtype=next(model.bert.parameters()).dtype
    ) 
    extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

    embedding_output, position_ids = model.bert.embeddings(
        input_ids=input_ids,
        position_ids=None,
        token_type_ids=token_type_ids,
        inputs_embeds=None,
    )
    
    rel_pos = _calculate_rel_pos(model, embedding_output, position_ids)

    embedding_output.retain_grad()

    encoder_outputs = model.bert.encoder(
                embedding_output,
                attention_mask=extended_attention_mask,
                split_lengths=None,
                rel_pos=rel_pos,
            )
    sequence_output = encoder_outputs[0]
    encoder_outputs[1][0].retain_grad()

    pooled_output = model.bert.pooler(sequence_output)
    pooled_output = model.dropout(pooled_output)
    scores = model.classifier(pooled_output)
    
    return _compute_attributions(scores, encoder_outputs[1][0])

def compute_bert_saliency_attributions(text, model, tokenizer):
    
    weights = model.bert.embeddings.word_embeddings
    weights_values = weights._parameters['weight'].cpu().data

    token_ids = tokenizer.encode(text, add_special_tokens=True, return_tensors="pt")
    token_ids_one_hot = F.one_hot(token_ids, num_classes=weights.num_embeddings)[0]
    token_ids_one_hot = token_ids_one_hot.float()

    embeddings = torch.mm(token_ids_one_hot, weights_values.cpu())
    embeddings = embeddings.reshape((1, embeddings.shape[0], embeddings.shape[1]))
    embeddings = torch.tensor(embeddings, requires_grad=True)
    
    scores = model(inputs_embeds=embeddings)[0]

    # Get the index corresponding to the maximum score and the maximum score itself.
    score_max_index = scores.argmax()
    score_max = scores[0, score_max_index]
    score_max.backward()
    saliency, _ = torch.max(embeddings.grad.data,dim=0)
    
    saliency_token = saliency.sum(dim=1)
    values = saliency_token.abs()/saliency_token.abs().sum()
    values = values.numpy()
    
    return values