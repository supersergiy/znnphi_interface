from six import iteritems
import h5py
import numpy as np
from codegen import generate_function, timeit, print_tensor_lines, print_tensor_part_lines
from layers import allocate_layer_lines, forward_layer_lines
from layers import generate_param_string
from layers import conv

def allocate_all_layers_lines(net, cores, ht):
    lines = []
    tensors, layer_info, layer_order, misc = net

    for (n,l) in iteritems(layer_info):
        lines += allocate_layer_lines(l, cores, ht)

    lines.append('')
    return lines

def allocate_featuremaps_lines(net):
    lines = []
    tensors, layer_info, layer_order, misc = net

    for (n,t) in iteritems(tensors):
        lines.append('tensors["{}"] = new znn::phi::hbw_array<float>(znn::phi::zero_init, {});'.format(n, t.memory_size))

    lines.append('')
    return lines

def generate_python_interface_misc(net):
    tensors, layer_info, layer_order, misc = net
    lines = []

    lines.append('input_size = {};'.format(tensors['user_input'].size))

    in_dim    = tensors['user_input'].dim
    out_dim   = tensors['output'].dim
    in_ndim   = len(in_dim)
    out_ndim  = len(out_dim)

    lines.append('in_dim  = {};'.format(in_ndim))
    lines.append('out_dim = {};'.format(out_ndim))

    lines.append('size_t tmp_in_shape[]  = {{ {} }};'.format(', '.join(map(str, in_dim))))
    lines.append('size_t tmp_out_shape[] = {{ {} }};'.format(', '.join(map(str, out_dim))))

    lines.append('in_shape.assign(tmp_in_shape, tmp_in_shape + {});'.format(in_ndim))
    lines.append('out_shape.assign(tmp_out_shape, tmp_out_shape + {});'.format(out_ndim))

    # Fill in the strides array for numpy output export
    out_strides = [4]  #sizeof float
    #go from the outer most dimension backward, then reverse
    for i in range(1, 5): #4 more dimensions
        out_strides.append(out_strides[-1] * out_dim[-i])
    out_strides = list(reversed(out_strides))

    lines.append('size_t tmp_out_strides[] = {{ {} }};'.format(', '.join(map(str, out_strides))))
    lines.append('out_strides.assign(tmp_out_strides, tmp_out_strides + {});'.format(out_ndim))
    lines.append('')

    return lines
def set_constants_lines():
    lines = []
    lines.append('this->lib_path = lib_path;')
    return lines

def constructor_body_lines(net, cores, ht):
    lines = []
    tensors, layer_info, layer_order, misc = net
    lines += set_constants_lines()
    lines += allocate_all_layers_lines(net, cores, ht)
    lines += allocate_featuremaps_lines(net)
    lines += generate_python_interface_misc(net)
    lines.append('')
    return lines

def forward_all_layers_lines(net):
    tensors, layer_info, layer_order, misc = net
    lines = []
    lines.append('std::cout << "Starting Forward Pass\\n";')
    count = 1
    for lname in layer_order:
       l = layer_info[lname]
       #lines.append('std::cout << "Running {}!\\n";'.format(l["name"]))
       #lines += forward_layer_lines(l)
       #lines += forward_layer_lines(l)
       #lines += timeit(forward_layer_lines(l), 1, l["name"] + ": ")
       #if l["type"] in ["pad"]:
       #lines += timeit(forward_layer_lines(l), 1, l["name"] + ": ")
       #lines.append("std::cout << \"{}\" << std::endl;".format(lname))
       lines += forward_layer_lines(l)

       #lines += print_tensor_part_lines(l["top"])
       #lines += print_tensor_lines(l["bot"])
       #lines += print_tensor_lines(l["top"])
       #lines.append('std::cout << "{} Finished!\\n";'.format(l["name"]))
       count += 1

    lines.append('')
    return lines


def forward_body_lines(net):
    tensors, layer_info, layer_order, misc = net
    lines = []
    lines += timeit(forward_all_layers_lines(net), 1, "average:")

    lines.append('')
    return lines

def generate_znet(net, out_path, cores, ht):
    lines = []
    #includes
    lines.append('#include <iostream>')
    lines.append('#include <chrono>')
    lines.append('#include <znn/layer/layers.hpp>')
    lines.append('#include <cstring>')
    lines.append('#include <znet.hpp>')
    lines.append('#include <common.hpp>')
    lines.append('')

    #constructor
    print "   Generating constructors..."
    constructor_signature = 'znn::phi::Znet::Znet(std::string weights_path, std::string lib_path)'
    constructor_body      = constructor_body_lines(net, cores, ht)
    constructor           = generate_function(constructor_signature, constructor_body)
    lines += constructor

    #forward pass
    print "   Generating foward pass..."
    forward_signature = 'void znn::phi::Znet::forward(void)'
    forward_body      = forward_body_lines(net)
    forward           = generate_function(forward_signature, forward_body)
    lines += forward

    f = open(out_path, 'w')
    for l in lines:
        f.write("{}\n".format(l))
