import torch
from copy import deepcopy
from ..rppo.init_model_gru import GRUAgent

def copy_model(model: GRUAgent, device):
    model_copy = deepcopy(model)
    return model_copy.to(device)

def load_model(path, state_dim, action_dim, device):
    model = GRUAgent(state_dim, action_dim, device).to(device)
    model_dict = torch.load(path, weights_only=True)
    model.load_state_dict(model_dict)
    return model