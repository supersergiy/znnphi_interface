from six import iteritems
import h5py
import numpy as np

from codegen import generate_function, zero_out_tensor, fill_tensor, timeit
from common import round_to_simd, block_bias, block_kernel

ACTIVATION = "true"

def generate_print_tensor(tname):
    lines = []
    lines.append('for (int i = 0; i < tensors["{}"]->num_elements(); i++) {{'.format(tname))
    lines.append('  cout << tensors["{}"]->data()[i] << " ";'.format(tname))
    lines.append('}')
    lines.append('std::cout << std::endl;')
    lines.append('')
    return lines

def generate_allocate_layers(net):
    lines = []
    tensors, layer_info, layer_order = net
    for (n,l) in iteritems(layer_info):
        if l["type"] == "conv":
            conv_params = "{}, {}, {}, {}, {}, {}, {}, {}, {}, {}".format(l["bn"], l["ifm"], l["ofm"],
                                                              l["id"], l["ihw"],
                                                              l["kernel_dim"][2], l["kernel_dim"][3],
                                                              l["pad"][0],  l["pad"][1], ACTIVATION)
            lines.append('layers["{}"] = new znn::phi::ConvWrapper({});'.format(l["name"],
                                                                                conv_params))
        elif l["type"] == "pool":
            bot_dim = tensors[l["bot"]].dim
            pool_params = ', '.join(['{}']*8).format( #8 parameters
                                                    bot_dim[0], bot_dim[1], bot_dim[2],
                                                    bot_dim[3], bot_dim[4],
                                                    l["kernel_dim"][0], l["kernel_dim"][1],
                                                    l["stride"][0], l["stride"][1])
            lines.append('layers["{}"] = new znn::phi::MaxPoolingLayer({});'.format(l["name"],
                                                                                pool_params))


        elif l["type"] == 'block_input':
            user_input = tensors["user_input"]
            block_params = '{}, {}, {}, {}'.format(user_input.dim[0],
                                                   user_input.dim[1],
                                                   user_input.dim[2],
                                                   user_input.dim[3])

            lines.append('layers["{}"] = new znn::phi::BlockDataLayer({});'.format(l["name"],
                                                                                block_params))
        elif l["type"] == 'unblock_output':
            user_output = tensors["user_output"]
            unblock_params = '{}, {}, {}, {}'.format(user_output.dim[0],
                                                     user_output.dim[1],
                                                     user_output.dim[2],
                                                     user_output.dim[3])

            lines.append('layers["{}"] = new znn::phi::UnblockDataLayer({});'.format(l["name"],
                                                                                unblock_params))
    lines.append('')
    return lines

def generate_initialize_weights(net, weights_path):
    lines = []
    tensors, layer_info, layer_order = net
    return lines
    #allocate_weights
    for (n,l) in iteritems(layer_info):
       if l["type"] in ["conv", "deconv"]:
          lines.append('tensors["{}"] = new znn::phi::hbw_array<float>({});'.format(
                                                      l["kernel"], l["kernel_size"]))

          lines.append('tensors["{}"] = new znn::phi::hbw_array<float>({});'.format(
                                                          l["bias"], l["bias_size"]))
    lines.append('')

    #initialize weights
    for (lname, l) in iteritems(layer_info):
        if l["type"] == "conv":
            if weights_path:
                weights = h5py.File(weights_path) ['data']
                lweights = weights[lname].values()
            else:
                print "WARNING: uninitialized layer {}".format(lname)
                lweights = []
                lweights.append(np.ones(l["kernel_dim"], dtype=np.float))
                lweights.append(np.zeros(l["bias_dim"], dtype=np.float))

            kernel = lweights[0][:]
            blocked_kernel = block_kernel(kernel, l)
            lines += fill_tensor('{}_kernel'.format(lname), blocked_kernel)

            if len(lweights) > 1:
                bias = lweights[1][:]
                print bias
                blocked_bias = block_bias(bias, l)
                print blocked_bias
                lines += fill_tensor('{}_bias'.format(lname), blocked_bias)
            else:
                lines += zero_out_tensor('{}_bias'.format(lname))

    lines.append('')
    return lines

def generate_allocate_featuremaps(net):
    lines = []
    tensors, layer_info, layer_order = net

    for (n,t) in iteritems(tensors):
       lines.append('tensors["{}"] = new znn::phi::hbw_array<float>({});'.format(n, t.memory_size))

    lines.append('')
    return lines

def generate_set_in_out_dimensions(net):
    tensors, layer_info, layer_order = net
    lines = []
    #input
    lines.append('input_size = {};'.format(tensors['user_input'].size))
    #output
    out_dim = 5
    out_strides = [4]  #sizeof float

    #go from the outer most dimension backward,
    #then reverse
    for i in range(1, 5): #4 more dimensions
        out_strides.append(out_strides[-1]*tensors['user_output'].dim[-i])
    out_strides = list(reversed(out_strides))

    lines.append('out_dim = {};'.format(out_dim))
    lines.append('size_t tmp_shape[] = {{ {} }};'.format(', '.join(map(str, tensors['user_output'].dim))))
    lines.append('out_shape.assign(tmp_shape, tmp_shape + {});'.format(out_dim))

    lines.append('size_t tmp_strides[] = {{ {} }};'.format(', '.join(map(str, out_strides))))
    lines.append('out_strides.assign(tmp_strides, tmp_strides + {});'.format(out_dim))
    return lines

def generate_constructor_body(net, weights_path):
    lines = []
    tensors, layer_info, layer_order = net

    lines += generate_allocate_featuremaps(net)
    lines += generate_initialize_weights(net, weights_path)
    lines += generate_allocate_layers(net)

    lines += generate_set_in_out_dimensions(net)

    lines.append('')


    return lines

'''def generate_load_data(net):
    tensors, layer_info, layer_order = net
    lines = []
    input_values = [1, 2, 3, 4]*(tensors['user_input'].size/4)
    lines += fill_tensor('user_input', input_values)
    return lines'''

def generate_forward_all_layers(net):
    tensors, layer_info, layer_order = net
    lines = []

    lines.append('std::cout << "Starting Forward Pass\\n";')
    for lname in layer_order:
       l = layer_info[lname]
       #lines.append('std::cout << "Running {}!\\n";'.format(l["name"]))

       if l["type"] in ["conv"]:
           params  = 'tensors["{}"]->data(), tensors["{}"]->data(), '.format(l["bot"], l["top"])
           params += 'tensors["{}"]->data(), tensors["{}"]->data()'.format(l["kernel"], l["bias"])
           #lines.append('layers["{}"]->forward({});'.format(lname, params))
       if l["type"] in ["pool", "block_input", "unblock_output"]:
           params  = 'tensors["{}"]->data(), tensors["{}"]->data(), '.format(l["bot"], l["top"])
           params += 'NULL, NULL'
           lines += (timeit(['layers["{}"]->forward({});'.format(lname, params)],
                            1, l["name"]+": "))

       #lines.append('std::cout << "{} Finished!\\n";'.format(l["name"]))
    lines.append('')
    return lines


def generate_forward_body(net):
    tensors, layer_info, layer_order = net
    lines = []

    #lines += generate_load_data(net)
    lines += timeit(generate_forward_all_layers(net), 1, "average:")
    return lines

def generate_znet(net, weights_path, out_path):
    '''tmp = net[1]["conv1_d1"]
    net[1].clear()
    net[1]["conv1_d1"] = tmp
    net[2].clear()
    net[2].append("conv1_d1")'''

    lines = []
    #includes
    lines.append('#include <iostream>')
    lines.append('#include <chrono>')
    lines.append('#include <znn/interface/conv_wrapper.hpp>')
    lines.append('#include <znn/layer/block_data.hpp>')
    lines.append('#include <znn/layer/pool/pool.hpp>')
    lines.append('#include <znn/layer/unblock_data.hpp>')
    lines.append('#include <cstring>')
    lines.append('#include <znet.hpp>')
    lines.append('#include <common.hpp>')
    lines.append('')

    #constructor
    constructor_signature = 'znn::phi::Znet::Znet(std::string weights_path)'
    constructor_body      = generate_constructor_body(net, weights_path)
    constructor           = generate_function(constructor_signature, constructor_body)
    lines += constructor

    #forward pass
    forward_signature = 'void znn::phi::Znet::forward(void)'
    forward_body      = generate_forward_body(net)
    forward           = generate_function(forward_signature, forward_body)
    lines += forward

    #write lines to file
    f = open(out_path, 'w')
    for l in lines:
        f.write("{}\n".format(l))


