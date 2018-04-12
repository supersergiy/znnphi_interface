import copy
from .common import generate_param_string

def set_dummy_data_dim(params, reference_tensor):
    top_dim = copy.copy(reference_tensor.dim)
    params["top_dim"] = top_dim

