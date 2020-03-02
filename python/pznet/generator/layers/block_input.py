import copy
from .common import generate_param_string

def allocate_block_input_lines(lparam):
    bot_dim = lparam["bot_dim"]

    block_param = bot_dim
    param_str = generate_param_string(block_param[0:4]) #assumes h=w
    lines = []
    lines.append('layers["{}"] = new znn::phi::BlockDataLayer({});'.format(
                                                                       lparam["name"],
                                                                       param_str))
    return lines


