import os
import torch


def reload_trial_model(model, trial_path, strict=True):
    model_path = os.path.join(trial_path, 'model.pth')
    model.load_state_dict(
        torch.load(model_path, map_location='cpu'), 
        strict=strict
    )
    
    
def parse_model_output(model_output, color_vocab, target_key='S0_probability', batch_index=0, topk=10, print_it=True):
    """ Parse a LUX model output to see color names and their probabilities
    
    Args: 
        model_output (dict): output of the model
        color_vocab (Vocabulary): the vocabulary from either XKCD or CIC
        target_key (str): [default='S0_probability'] the key in the model output dict to use
        batch_index (int): [default=0] the batch item to parse
        topk (int): [default=10] the number of color names to parse
        print_it (bool): [default=True] if False, a dictionary that maps color names to probabilities
            will be output
    """
    if not print_it:
        out = {}
    
    top_k = model_output[target_key][batch_index].topk(k=topk, dim=0)    
    for index, proba in zip(top_k.indices, top_k.values):
        color_name = color_vocab.lookup_index(index.item())
        if print_it:
            print(f'P({color_name:^20}|x)\t={proba:0.4f}')
        else:
            out[color_name] = proba
    if not print_it:
        return out