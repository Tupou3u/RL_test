from algs.dqn.net import *

def copy_model(model, device):
    if isinstance(model, QNetwork):
        copy_model = QNetwork(model.state_dim, model.action_dim)
        copy_model.load_state_dict(model.state_dict())
        return copy_model.to(device)