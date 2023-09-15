import re
import torch
import numpy as np

def tf_wrename(w_name):
    return w_name.split(':')[0].replace('kernel','weight').replace('/','.')

## Get_weights
def get_torch_weights(model):
    return {w_name: np.transpose(weight) for w_name, weight in model.state_dict().items()}

def get_tf_weights(model):
    return {tf_wrename(w.name): w.numpy() for w in model.trainable_variables }

## Set_weights
def set_torch_weights(model, weights):
  torch_state_dict = {w_name: torch.from_numpy(weight) for w_name, weight in weights.items()}
  model.load_state_dict(torch_state_dict, strict=False)

def set_tf_weights(model, weights):
  assigned_weights = []
  weight_list = [ w.numpy() for w in model.weights ]
  weight_name_list = [tf_wrename(w.name) for w in model.weights ]
  for w_name, weight in weights.items():
    if w_name in weight_name_list:
      index = weight_name_list.index(w_name)
      assigned_weights.append(w_name)
      if weights[w_name].shape == weight_list[index].shape:
        weight_list[index] = weights[w_name]
      else:
        weight_list[index] = np.transpose(weights[w_name])
  model.set_weights(weight_list)
  return assigned_weights

## W_TF2Torch
def W_TF2Torch(tf_model, torch_model):
    tf_weights = get_tf_weights(tf_model)
    torch_weights = get_torch_weights(torch_model)
    assigned_weights = set_torch_weights(torch_model, tf_weights)
    print('missing_tf_weights : ', [name for name in tf_weights.keys() if name not in assigned_weights])
    print('missing_torch_weights : ', [name for name in torch_weights.keys() if name not in assigned_weights])

def W_Torch2TF(torch_model, tf_model):
    tf_weights = get_tf_weights(tf_model)
    torch_weights = get_torch_weights(torch_model)
    assigned_weights = set_tf_weights(tf_model, torch_weights)
    print('missing_tf_weights : ', [name for name in tf_weights.keys() if name not in assigned_weights])
    print('missing_torch_weights : ', [name for name in torch_weights.keys() if name not in assigned_weights])