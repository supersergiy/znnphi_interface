import copy
from common import generate_param_string

def allocate_unblock_output_lines(lparam):
    bot_dim = lparam["bot_dim"]

    unblock_param = bot_dim
    param_str = generate_param_string(unblock_param[0:4]) #assuems h=w
    lines = []
    lines.append('layers["{}"] = new znn::phi::UnblockDataLayer({});'.format(
                                                                       lparam["name"],
                                                                       param_str))
    return lines


