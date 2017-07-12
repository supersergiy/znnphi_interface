#!/usr/bin/python
import sys
import os
from subprocess import  STDOUT, check_call

def parse_args():
    args = {}

    for i in range(1, len(sys.argv)):
        a = sys.argv[i]
        k, v = a.split('=')
        args[k.lower()] = v

    return args

def get_out_name(args):
    name = '' 
    name += args["layer"]
    
    for k in sorted(args.keys()):
        if k != "layer":
            name += '_' + args[k]
    
    name += ".so"
    return name

def create_wrapper(args):
    if args["layer"] == "conv":
        return create_conv_wrapper(args)
    else:
        raise Exception("Not implemented layer")

def create_conv_wrapper(args):
    activation = "true" if args["activation"] else "false"
    add_or_overwrite = "true" if args["addoroverwrite"] else "false"

    template_params = ", ".join([ args["cores"], args["ht"],
                                  args["bn"], args["ifm"], args["ofm"], 
                                  args["id"], args["ihw"], args["kd"], args["khw"], 
                                  args["padd"], args["padhw"],
                                  activation, add_or_overwrite ]) 
    pid = os.getpid()
    wrapper_name = "./.tmp/{}.cpp".format(pid) 
    with open("./wrapper_base.cpp", 'r') as in_f:
        with open(wrapper_name, 'w') as out_f:
            for l in in_f: 
                out_f.write(l.replace("[LAYER_NAME]",      "Conv").
                              replace("[TEMPLATE_PARAMS]", template_params))

    return wrapper_name 

def compile_dl(args):
    out_name = get_out_name(args)
    out_path = "../../../../lib/{}".format(out_name)

    wrapper_name = create_wrapper(args)
    print "./" + out_path

    with open(os.devnull, 'wb') as devnull:
        check_call(['make', wrapper_name.replace(".cpp", ".so"), "O={}".format(out_path)], stdout=devnull, stderr=STDOUT)
        check_call(['rm', "-f", wrapper_name], stdout=devnull, stderr=STDOUT)

def main():
    args = {}
    args = parse_args()
    compile_dl(args)
    
if __name__ == "__main__":
    main()
