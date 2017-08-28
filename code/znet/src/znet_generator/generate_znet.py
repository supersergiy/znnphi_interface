from six import iteritems
import h5py
import numpy as np
from codegen import generate_function, timeit, print_tensor_lines, print_tensor_part_lines
from layers import allocate_layer_lines, forward_layer_lines
from layers import generate_param_string

def allocate_all_layers_lines(net):
    lines = []
    tensors, layer_info, layer_order, misc = net

    for (n,l) in iteritems(layer_info):
        lines += allocate_layer_lines(l)

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

    lines.append('')
    return lines

def constructor_body_lines(net):
    lines = []
    tensors, layer_info, layer_order, misc = net

    lines += allocate_featuremaps_lines(net)
    lines += allocate_all_layers_lines(net)
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
       #lines += timeit(forward_layer_lines(l), 1, l["name"] + ": ")
       #if l["type"] in ["pad"]:
       lines += timeit(forward_layer_lines(l), 1, l["name"] + ": ")
       #lines += forward_layer_lines(l)

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

def generate_znet(net, out_path):
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
    constructor_signature = 'znn::phi::Znet::Znet(std::string weights_path)'
    constructor_body      = constructor_body_lines(net)
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

def generate_template_znet(net, out_path):
    lines = []
    #includes
    lines.append('#include "znn/bench/forward2.hpp"')
    lines.append('')
    lines.append('using namespace znn::phi;')
    lines.append('')
    tensors, layer_info, layer_order, misc = net
    #main
    main_signature = 'int main(void)' 
    main_body      = [] 
    for lname in layer_order: 
        l = layer_info[lname]
        if l["type"] == "conv":
            params = [l["bn"], l["ifm"], l["ofm"], l["id"], l["ihw"], 
                      l["json_kernel_size"][0], l["json_kernel_size"][1], 
                      l["pad"][0], l["pad"][1]]
            param_str = generate_param_string(params)
            main_body.append('benchmark_forward<{}>("{}");'.format(param_str, l["name"]))

    main = generate_function(main_signature, main_body)
    lines += main

    f = open(out_path, 'w')
    for l in lines:
        f.write("{}\n".format(l))


