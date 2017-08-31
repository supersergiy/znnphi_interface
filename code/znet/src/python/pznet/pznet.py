from common import convert_prototxt_to_json
import os
import sys
import inspect
import time

class znet:
    #TODO: create temporary folder on module startup,
    #      put all the .so's and stuff in it, delete on exit

    #FOR NOW we assume that we always have only one network,
    #so we can work inside the repo like real scrubs

    def __init__(self):
        self.net = None
        self.real_secret_path = "/home/ubuntu/znnphi_interface/code/znet/src/python/pznet/.tmp/"        
        sys.path.append(self.real_secret_path)

    def create_net(self, prototxt_path, h5_weights_path, output_path, cores):
        json_net_path = os.path.join(self.real_secret_path, 'net.json')
        convert_prototxt_to_json(prototxt_path, json_net_path)
        
        znnphi_path = os.environ["ZNNPHI_PATH"]
        mothership_folder = '{}/code/znet'.format(znnphi_path)

        make_command  = 'make -C {} py N={} W={} O={} CORES={}'.format(mothership_folder,
                                                         json_net_path,
                                                         h5_weights_path,
                                                         self.real_secret_path,
                                                         cores)
        os.system(make_command) #compiles the znet.so and copies it to the working folder along with the weights

        #copy results to the output folder
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        os.system("cp -r {}/* {}".format(self.real_secret_path, output_path))

    def load_net(self, path_to_net):
        os.system("cp -r {}/* {}".format(path_to_net, self.real_secret_path))
        import znet
        self.net = znet.znet(os.path.join(self.real_secret_path, "weights/"))

    def forward(self, input_tensor):
        return self.net.forward(input_tensor)
