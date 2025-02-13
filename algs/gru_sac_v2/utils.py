from copy import deepcopy

def copy_model(model, device):
    copy_model = deepcopy(model)
    return copy_model.to(device)