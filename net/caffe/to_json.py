#!/bin/python
from google.protobuf.descriptor import FieldDescriptor
from google.protobuf.text_format import Merge
from protobuf_to_dict import protobuf_to_dict, TYPE_CALLABLE_MAP
import caffe_pb2
import json
import sys

net = caffe_pb2.NetParameter()

prototxt_path = sys.argv[1]
Merge((open(prototxt_path, 'r')).read(), net)

with open("test.json", 'wb') as f:
    json.dump(protobuf_to_dict(net), f, indent=3)



