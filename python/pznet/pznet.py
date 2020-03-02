from .common import convert_prototxt_to_json
import os
import sys
import inspect
import time
from copy import deepcopy

TMP_DIR = "/opt/.tmp"

class znet:
    #TODO: create temporary folder on module startup,
    #      put all the .so's and stuff in it, delete on exit

    #FOR NOW we assume that we always have only one network,
    #so we can work inside the repo like real scrubs

    def __init__(self, net_path):
        lib_path = os.path.join(net_path, "lib")
        if not os.path.exists(lib_path):
            os.makedirs(lib_path)

        sys.path.append(net_path)
        try:
            import znet
        except:
            raise Exception("Problem loading the network object. " +
                            "Please make sure there's a znet.so file present at {}".format(net_path))

        self.net = znet.znet(os.path.join(net_path, "weights/"), lib_path)
    
    @classmethod
    def from_pytorch_model(self, pytorch_model, output_path,
                     architecture='AVX2', core_options={'conv': [2, 2]},
                     cpu_offset=0, opt_mode='full_opt', ignore='', 
                     time_each=False, python_v=3):
        """
        Load and compile pytorch model to znet.so file with weights.
        The compiled files will be packed inside a zip file.
        """

        znnphi_path = os.environ["ZNNPHI_PATH"]
        mothership_folder = os.path.join(znnphi_path, 'code/znet')

        final_cores = deepcopy(core_options)
        if 'act' not in final_cores or final_cores['act'][0] < 0:
            final_cores['act'][0] = final_cores['conv'][0]
        if 'act' not in final_cores or final_cores['act'][1] < 0:
            final_cores['act'][1] = final_cores['conv'][1]
        if 'lin' not in final_cores or final_cores['lin'][0] < 0:
            final_cores['lin'][0] = final_cores['conv'][0]
        if 'lin' not in final_cores or final_cores['lin'][1] < 0:
            final_cores['lin'][1] = final_cores['conv'][1]
        
        # write model as json format
        with open(path_to_json, 'w', encoding="utf8") as f:
            net_dict = net.parameters()
            json.dump(net_dict, f, indent=3)
 
        #compiles the znet.so and copies it to the working folder along with the weights
        make_command = f'make -C {mothership_folder} py N={json_net_path} W={h5_weights_path}' +\
            f' O={self.tmp_folder_path} ARCH={architecture} CONV_CORES={final_cores["conv"][0]} ' +\
            f'CONV_HT={final_cores["conv"][1]} ACT_CORES={final_cores["act"][0]} ' +\
            f'ACT_HT={final_cores["act"][1]} LIN_CORES={final_cores["lin"][0]} ' +\
            f'LIN_HT={final_cores["lin"][1]} CPU_OFFSET={cpu_offset} PZ_OPT={opt_mode} ' +\
            f'IGNORE={ignore} TIME_EACH={time_each} PYTHON_V={python_v}'
        os.system(make_command)

        #copy results to the output folder
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        os.system("cp -r  {}/* {}".format(self.tmp_folder_path, output_path))
        os.system("rm -rf {}/*   ".format(self.tmp_folder_path))
        os.system("cp {} {}/net.prototxt".format(prototxt_path, output_path))
        os.system("cp {} {}/weights.h5".format(h5_weights_path, output_path))
        
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
