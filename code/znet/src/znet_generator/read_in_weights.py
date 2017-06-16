import h5py 
from six import iteritems

def read_in_weights(net, weights_path):
    lines = []
    tensors, layer_info, layer_order = net

    #initialize weights
    weights = h5py.File(weights_path) ['data']

    for (lname, l) in iteritems(layer_info):
        if l["type"] == "conv":
            lweights = weights[lname].values()
            l["kernel_data"] = lweights[0][:]

            if len(lweights) > 1:
                l["bias_data"] = lweights[1][:]
            else:
                l["bias_data"] = None
        elif l["type"] == "bnorm":
            lweights = weights[lname].values()
            l["scale_data"] = lweights[0][:]
            l["bias_data"]  = lweights[1][:]
            if len(lweights) > 2:
                scale_factor = lweights[2][:]
                if scale_factor.size != 1:
                    import pdb; pdb.set_trace()
                    raise Exception("wtf")
                l["scale_data"] *= scale_factor 
                l["bias_data"]  *= scale_factor 

    lines.append('')
    return lines


