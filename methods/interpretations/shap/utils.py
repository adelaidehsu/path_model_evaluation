from shap import maskers

class callable_tokenizer():
    
    def __init__(self, tokenizer):
        self.tokenizer=tokenizer
        
    def encode(self, x):
        return self.tokenizer.encode(x)
    
    def encode_plus(self, x):
        return self.tokenizer.encode_plus(x, add_special_tokens=True, 
                                         max_length=512,
                                         truncation=True, 
                                         padding = "max_length", 
                                         return_attention_mask=True)
    
    def __call__(self, x):
        return self.tokenizer.encode_plus(x, add_special_tokens=True, 
                                         max_length=512,
                                         truncation=True, 
                                         padding = "max_length", 
                                         return_attention_mask=True)

def convert_to_masker(tokenizer):
    
    new_tokenizer = callable_tokenizer(tokenizer)

    for attr_str in dir(tokenizer):
        if attr_str not in ['__class__']:
            attr = getattr(tokenizer, attr_str)
            try:
                setattr(new_tokenizer, attr_str, attr)
            except:
                continue

    new_tokenizer.tokenizer=tokenizer
    return maskers.Text(new_tokenizer)