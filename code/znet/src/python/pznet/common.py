#!/bin/python
from google.protobuf.descriptor import FieldDescriptor
from google.protobuf.text_format import Merge
from protobuf_to_dict import protobuf_to_dict, TYPE_CALLABLE_MAP

import caffe_pb2
import json

def convert_prototxt_to_json(path_to_prototxt, path_to_json):
    net = caffe_pb2.NetParameter()
    Merge((open(path_to_prototxt, 'r')).read(), net)

    with open(path_to_json, 'wb') as f:
        json.dump(protobuf_to_dict(net), f, indent=3)



