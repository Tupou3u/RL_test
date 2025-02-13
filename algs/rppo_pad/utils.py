from .init_model_gru import GRUAgent
import torch
from copy import deepcopy

def copy_model(model: GRUAgent, device):
    model_copy = deepcopy(model)
    model_copy.load_state_dict(model.state_dict())
    return model_copy.to(device)

def load_model(path, state_dim, action_dim, device):
    model = GRUAgent(state_dim, action_dim, device).to(device)
    model_dict = torch.load(path, weights_only=True)
    model.load_state_dict(model_dict)
    return model