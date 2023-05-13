import torch

from captum.attr import visualization as viz

def compute_input_type_attention(text, tokenizer):    
    input_ids = tokenizer.encode(text, add_special_tokens=True, max_length=512)
    input_ids = torch.tensor([input_ids])

    attention_mask = construct_attention_mask(input_ids)

    token_type_ids = construct_input_token_type(input_ids)
    
    return input_ids, token_type_ids, attention_mask

def construct_attention_mask(input_ids):
    return torch.ones_like(input_ids)

def construct_input_token_type(input_ids, sep_ind=0):
    seq_len = input_ids.size(1)
    token_type_ids = torch.tensor([[0 if i <= sep_ind else 1 for i in range(seq_len)]])
    return token_type_ids

def visualize(attributions, text, label, model, tokenizer):
    model.cpu()
    input_ids, token_type_ids, attention_mask = compute_input_type_attention(text, tokenizer)
    all_tokens = tokenizer.convert_ids_to_tokens(input_ids[0].detach().tolist())

    vis = viz.VisualizationDataRecord(
                            word_attributions=attributions*5,
                            pred_prob=torch.max(torch.softmax(model(input_ids, token_type_ids=token_type_ids)[0][0],dim=0)),
                            pred_class=torch.argmax(model(input_ids, token_type_ids=token_type_ids)[0][0]),
                            true_class=label,
                            attr_class=label,
                            attr_score=attributions.sum(),
                            raw_input=all_tokens,
                            convergence_score=0.5)

    viz.visualize_text([vis])