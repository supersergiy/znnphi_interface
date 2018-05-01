#!/bin/python
from google.protobuf.descriptor import FieldDescriptor
from google.protobuf.text_format import Merge
from .protobuf_to_dict import protobuf_to_dict, TYPE_CALLABLE_MAP
from . import znet_caffe_pb2

import json

def convert_prototxt_to_json(path_to_prototxt, path_to_json):
    net = znet_caffe_pb2.NetParameter()
    Merge((open(path_to_prototxt, 'r')).read(), net)

    with open(path_to_json, 'w') as f:
        net_dict = protobuf_to_dict(net)
        json.dump(net_dict, f, indent=3)

if __name__ == "__main__":
    import sys
    convert_prototxt_to_json(sys.argv[1], sys.argv[2])
