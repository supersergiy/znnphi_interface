import copy
from .common import generate_param_string

def allocate_pad_lines(lparam):
    l = lparam
    allocation_params = [lparam["bn"], lparam["ifm"], lparam["id"], lparam["ihw"], lparam["padd"], lparam["padhw"]]

    param_str = generate_param_string(allocation_params)
    lines = []
    #allocate layer
    lines.append('layers["{}"] = new znn::phi::PadLayer({});'.format(l["name"],
                                                                        param_str))
    return lines

