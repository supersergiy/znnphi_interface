import os
import sys
import inspect
import time
from copy import deepcopy

from tempfile import mkdtemp
TMP_DIR = mkdtemp()

from .common import convert_prototxt_to_json


class PZNet:
    def __init__(self, net_path, lib_path=None):
        if lib_path is None:
            lib_path = os.path.join(net_path, "lib")

        if not os.path.exists(lib_path):
            os.makedirs(lib_path)
        
        sys.path.append(net_path)
        try:
            import znet
        except:
            raise Exception(
                "Problem loading the network object. " +
                "Please make sure there's a znet.so file" +
                " present at {}".format(net_path)
            )
        
        self.net = znet.znet(os.path.join(net_path, "weights/"), lib_path)
    
    @classmethod
    def from_kaffe_model(
        cls, prototxt_path, h5_weights_path, output_net_path,
        architecture='AVX2', core_options={'conv': [2, 2]},
        cpu_offset=0, opt_mode='full_opt', ignore='', 
        time_each=False):
        """
        The compiled files will be packed inside a zip file.
        """
        if not os.path.exists(output_net_path):
            os.makedirs(output_net_path)
       
        CPP_FOLDER = os.path.join(os.path.dirname(__file__), '../../cpp')

        final_cores = deepcopy(core_options)
        if 'act' not in final_cores:
            final_cores['act'] = final_cores['conv']
        else:
            if final_cores['act'][0] < 0:
                final_cores['act'][0] = final_cores['conv'][0]
            if final_cores['act'][1] < 0:
                final_cores['act'][1] = final_cores['conv'][1]

        if 'lin' not in final_cores:
            final_cores['lin'] = final_cores['conv']
        else:
            if final_cores['lin'][0] < 0:
                final_cores['lin'][0] = final_cores['conv'][0]
            if final_cores['lin'][1] < 0:
                final_cores['lin'][1] = final_cores['conv'][1]
        
        # write model as json format
        json_net_path = os.path.join(output_net_path, 'model.json')
        convert_prototxt_to_json(prototxt_path, json_net_path)

        #compiles the znet.so and copies it to the working folder along with the weights
        tmp_folder_path = '/tmp/pznet'
        make_command = f'make -C {CPP_FOLDER} py N={json_net_path} W={h5_weights_path} O={tmp_folder_path} ARCH={architecture} CONV_CORES={final_cores["conv"][0]} CONV_HT={final_cores["conv"][1]} ACT_CORES={final_cores["act"][0]} ACT_HT={final_cores["act"][1]} LIN_CORES={final_cores["lin"][0]} LIN_HT={final_cores["lin"][1]} CPU_OFFSET={cpu_offset} PZ_OPT={opt_mode} IGNORE={ignore} TIME_EACH={time_each}'
        print('make command: \n\n', make_command, '\n')
        os.system(make_command)

        #copy results to the output folder
        os.system("cp -r  {}/* {}".format(tmp_folder_path, output_net_path))
        os.system("rm -rf {}".format(tmp_folder_path))
        os.system("cp {} {}/net.prototxt".format(prototxt_path, output_net_path))
        os.system("cp {} {}/weights.h5".format(h5_weights_path, output_net_path))
        return cls(output_net_path) 
    
    @property
    def in_shape(self):
        ret = self.net.get_in_shape()
        return ret

    @property
    def out_shape(self):
        ret = self.net.get_out_shape()
        return ret

    def forward(self, input_tensor):
        return self.net.forward(input_tensor)
