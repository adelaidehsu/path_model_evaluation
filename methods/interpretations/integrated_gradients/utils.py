import torch
import torch.nn.functional as F

def forward_with_softmax(inputs, token_type_ids, attention_mask, model=None):
    output = model(inputs, token_type_ids=token_type_ids, attention_mask=attention_mask, position_ids=None)
    return F.softmax(output[0])

def summarize_attributions(attributions):
    attributions = attributions.sum(dim=-1).squeeze(0)
    attributions = attributions / torch.norm(attributions)
    return attributions