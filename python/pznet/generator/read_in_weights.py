import h5py
from six import iteritems
import numpy as np
import copy

def read_in_weights(net, weights_path):
    lines = []
    tensors, layer_info, layer_order, misc = net

    #initialize weights
    weights = h5py.File(weights_path, 'r')['data']
    for (lname, l) in iteritems(layer_info):
        if l["type"] in ["conv", "deconv"]:
            lweights = weights[lname]
            l["kernel_data"] = lweights['0'][:]

            if len(lweights) > 1:
                l["bias_data"] = lweights['1'][:]
            else:
                l["bias_data"] = None
        elif l["type"] in ["scale"]:
            lweights = weights[lname]

            l["scale_data"] = copy.deepcopy(lweights['0'][:])
            l["bias_data"]  = copy.deepcopy(lweights['1'][:])
            l["scale_data"].resize(l["scale_size"])
            l["bias_data"].resize(l["bias_size"])

            if len(lweights) > 2:
                scale_factor = lweights['2']
                if scale_factor.size != 1:
                    raise Exception("wtf")
                l["scale_data"] *= scale_factor
                l["bias_data"]  *= scale_factor
        elif l["type"] in ["bias"]:
            lweights = weights[lname]

            l["bias_data"]  = lweights['0'][:]
            l["scale_data"] = np.ones(l["bias_data"].shape)
            l["type"] = "scale" # hashtag polymorph

            if len(lweights) > 2:
               import pdb; pdb.set_trace()

        elif l["type"] in ["bnorm"]:
            lweights = weights[lname]

            mean_data = lweights['0'][:]
            var_data  = lweights['1'][:]

            if len(lweights) > 2:
                scale_factor = lweights['2']
                if scale_factor.size != 1:
                    import pdb; pdb.set_trace()
                    raise Exception("wtf")
                if (scale_factor != 0):
                    mean_data /= scale_factor
                    var_data  /= scale_factor

            var_data += 0.00000001
            std_data  = np.sqrt(var_data)

            l["bias_data"]  = -1.0*np.divide(mean_data, std_data)
            l["scale_data"] = 1.0  / std_data

            l["scale_data"].resize(l["scale_size"])
            l["bias_data"].resize(l["bias_size"])


    lines.append('')
    return lines
