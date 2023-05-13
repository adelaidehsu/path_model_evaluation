import torch.nn.functional as F

def predictor(texts):
    probas = []
    n = len(texts)
    BATCH_SIZE = 4
    num_batches = n//BATCH_SIZE
        
    probas = []
        
    for i in range(num_batches):
        encoding = tokenizer.batch_encode_plus(texts[i*BATCH_SIZE:(i+1)*BATCH_SIZE], 
                                         add_special_tokens=True, 
                                         max_length=512,
                                         truncation=True, 
                                         padding = "max_length", 
                                         return_attention_mask=True, 
                                         pad_to_max_length=True,
                                         return_tensors="pt")
        encoding = encoding.to('cuda')
        outputs = model(**encoding)
        proba = F.softmax(outputs[0]).detach().cuda().cpu().numpy()
    
        probas.append(proba)

    return np.concatenate(probas)