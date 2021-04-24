import numpy as np
from collections import defaultdict


def detach_tensor(tensor):
    if type(tensor) != np.ndarray:
        if type(tensor) == list:
            return np.ndarray(tensor)
        else:
            return tensor.cpu().detach().numpy()
    return tensor

def recursive_append(target_dict, source_dict):
    for e in source_dict:
        if type(source_dict[e]) == dict:
            if e not in target_dict:
                target_dict[e] = defaultdict(list)
            target_dict[e] = recursive_append(target_dict[e], source_dict[e])
        elif source_dict[e] is not None:
            if type(source_dict[e]) == list:
                target_dict[e].append(source_dict[e])
            else:
                target_dict[e].append(source_dict[e].cpu())
    
    return target_dict

def recursive_concat(source_dict):
    for e in source_dict:
        if type(source_dict[e]) == dict or type(source_dict[e]) == defaultdict:
            source_dict[e] = recursive_concat(source_dict[e])
        elif source_dict[e] is not None:
            source_dict[e] = np.concatenate(source_dict[e])
    
    return source_dict