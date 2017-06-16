from common import round_to_simd
from operator import mul

class Tensor:
    def __init__(self, dim):
        self.dim  = dim

        self.size = reduce(mul, dim)

        self.memory_size = 1

        self.memory_size *= dim[0]
        self.memory_size *= round_to_simd(dim[1])
        self.memory_size *= dim[2]
        self.memory_size *= dim[3]
        self.memory_size *= dim[4]
        
        self.who_touched_me = []
