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
        

    def create_net(self, prototxt_path, h5_weights_path, output_path):
        self.working_folder = output_path

        if not os.path.exists(self.working_folder):
            os.makedirs(self.working_folder)

        sys.path.append(self.working_folder)

        json_net_path = os.path.join(self.working_folder, 'net.json')
        convert_prototxt_to_json(prototxt_path, json_net_path)
        
        znnphi_path = os.environ["ZNNPHI_PATH"]
        mothership_folder = '{}/code/znet'.format(znnphi_path)

        make_command  = 'make -C {} py N={} W={} O={}'.format(mothership_folder,
                                                         json_net_path,
                                                         h5_weights_path,
                                                         self.working_folder)
        os.system(make_command) #compiles the znet.so and copies it to the working folder along with the weights

    def load_net(self, path_to_net):
        sys.path.append(path_to_net)
        import znet
        self.net = znet.znet(os.path.join(path_to_net, "weights/"))

    def forward(self, input_tensor):
        return self.net.forward(input_tensor)
