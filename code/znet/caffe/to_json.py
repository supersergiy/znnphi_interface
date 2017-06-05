#!/bin/python
from copy import copy
from google.protobuf.descriptor import FieldDescriptor
from google.protobuf.text_format import Merge
from protobuf_to_dict import protobuf_to_dict, TYPE_CALLABLE_MAP
import caffe_pb2
import json

net = caffe_pb2.NetParameter()
Merge((open("deploy.prototxt", 'r')).read(), net)

with open("test.json", 'wb') as f:
    json.dump(protobuf_to_dict(net), f, indent=3)



