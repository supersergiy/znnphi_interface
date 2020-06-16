#!/usr/bin/env python
import sys
import os
from tempfile import mkstemp
from subprocess import  STDOUT, check_call
import subprocess

my_path = sys.path[0]
#_, wrapper_path = mkstemp(suffix='.cpp')
temp_dir = '/tmp/pznet_jit'
#temp_dir = os.path.join(my_path, 'tmp')
if not os.path.isdir(temp_dir):
    os.mkdir(temp_dir)
else:
    os.remove(os.path.join(temp_dir, '*'))

wrapper_path = f'{temp_dir}/{os.getpid()}.cpp'

def parse_args():
    args = {}

    for i in range(1, len(sys.argv)):
        a = sys.argv[i]
        k, v = a.split('=')
        args[k.lower()] = v

    return args

def get_out_path(args):
    name = ''
    name += os.path.join(args["lib_folder"], args["layer"])

    for k in sorted(args.keys()):
        if k not in ["layer", "lib_folder"]:
            name += '_' + args[k]

    name += ".so"
    
    return name

def create_wrapper(args):
    if args["layer"] == "conv":
        return create_conv_wrapper(args)
    else:
        raise Exception("Not implemented layer")

def create_conv_wrapper(args):
    template_params = ", ".join(
        [args["cores"], args["ht"],
         args["bn"], args["ifm"], args["ofm"],
         args["id"], args["ihw"], args["kd"], args["khw"],
         args["out_d_skip"], args["out_padd"],
         args["out_h_skip"], args["out_w_skip"], args["out_padhw"],
         args["out_stride_d"], args["out_stride_hw"],
         args["activation"], args["addoroverwrite"], args["cpu_offset"]])

    with open(os.path.join(my_path, 'wrapper_base.cpp'), 'r') as in_f:
        with open(wrapper_path, 'w') as out_f:
            for l in in_f:
                out_f.write(l.replace("[LAYER_NAME]",      "Conv").
                              replace("[TEMPLATE_PARAMS]", template_params))

def compile_dl(args):
    out_path = get_out_path(args)

    create_wrapper(args)
    target_name = wrapper_path.replace(".cpp", ".so")
    # this print will be captured in the jit.hpp, 
    # so we can not add or modify print in this script!
    print(out_path)

    compile_command = f"make -C {my_path} {target_name} O={out_path} ARCH={args['arch']}"
    #compile_command  += ' 2> /dev/null'
    os.system(compile_command)
    check_call(['rm', "-f", wrapper_path], stderr=STDOUT)

def main():
    args = {}
    args = parse_args()
    compile_dl(args)

if __name__ == "__main__":
    main()
