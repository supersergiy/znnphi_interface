from common import convert_prototxt_to_json
import os
import sys
import inspect
import time

TMP_DIR = "/opt/.tmp"

class znet:
    #TODO: create temporary folder on module startup,
    #      put all the .so's and stuff in it, delete on exit

    #FOR NOW we assume that we always have only one network,
    #so we can work inside the repo like real scrubs

    def __init__(self):
        self.net = None
        my_path = os.path.dirname(os.path.abspath(__file__))
        self.tmp_folder_path = os.path.join(my_path, ".tmp/")

        if not os.path.exists(self.tmp_folder_path):
            os.makedirs(self.tmp_folder_path)

        sys.path.append(self.tmp_folder_path)

    def create_net(self, prototxt_path, h5_weights_path, output_path, architecture='AVX2', cores=2, ht=2, cpu_offset=0, opt_flags='all'):
        json_net_path = os.path.join(self.tmp_folder_path, 'net.json')
        convert_prototxt_to_json(prototxt_path, json_net_path)

        znnphi_path = os.environ["ZNNPHI_PATH"]
        mothership_folder = '{}/code/znet'.format(znnphi_path)

        make_command  = 'make -C {} py N={} W={} O={} ARCH={} CORES={} HT={} CPU_OFFSET={} PZ_OPT={}'.format(mothership_folder,
                                                             json_net_path,
                                                             h5_weights_path,
                                                             self.tmp_folder_path,
                                                             architecture, cores, ht, cpu_offset, opt_flags)
        os.system(make_command) #compiles the znet.so and copies it to the working folder along with the weights

        #copy results to the output folder
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        os.system("cp -r  {}/* {}".format(self.tmp_folder_path, output_path))
        os.system("rm -rf {}/*   ".format(self.tmp_folder_path))
        os.system("cp {} {}/net.prototxt".format(prototxt_path, output_path))
        os.system("cp {} {}/weights.h5".format(h5_weights_path, output_path))

    def load_net(self, net_path, lib_path=None):
        if lib_path is None:
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

    def get_in_shape(self):
        ret = self.net.get_in_shape()
        return ret

    def get_out_shape(self):
        ret = self.net.get_out_shape()
        return ret

    def forward(self, input_tensor):
        return self.net.forward(input_tensor)
