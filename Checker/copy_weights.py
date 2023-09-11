import re
import torch
import numpy as np

def tf_wrename(w_name):
    return re.search(r'/(.*?):', w_name).group(1).replace('kernel','weight')

## Get_weights
def get_torch_weights(model):
    weights = {}
    for layer_name, layer in model.named_children():
        layer_weights = {w_name: np.transpose(weight.detach().numpy())
                        for w_name, weight in layer._parameters.items()}
        if layer_weights:
            weights[layer_name] = layer_weights
    return weights

def get_tf_weights(model):
    weights = {}
    for layer in model.layers:
        layer_weights = {tf_wrename(w.name): w.numpy() for w in layer.weights}
        if layer_weights:
            weights[layer.name] = layer_weights
    return weights

## Set_weights
def set_torch_weights(model, weights):
    torch_state_dict = {f"{layer_name}.{w_name}": torch.from_numpy(np.transpose(weight))
                        for layer_name, layer_weights in weights.items()
                        for w_name, weight in layer_weights.items()}
    model.load_state_dict(torch_state_dict, strict=False)
    # print(f'Moved weight {torch_state_dict.keys()}')

def set_tf_weights(model, weights):
    for layer_name, layer_weights in weights.items():
        layer = model.get_layer(layer_name)
        list_w_name = [tf_wrename(w.name) for w in layer.weights]
        list_weight = [layer_weights[w_name] for w_name in list_w_name]
        layer.set_weights(list_weight)
        # print(f'Moved weight {list_w_name}')

## W_TF2Torch
def W_TF2Torch(tf_model, torch_model):
    tf_weights = get_tf_weights(tf_model)
    set_torch_weights(torch_model, tf_weights)

def W_Torch2TF(torch_model, tf_model):
    torch_weights = get_torch_weights(torch_model)
    set_tf_weights(tf_model, torch_weights)