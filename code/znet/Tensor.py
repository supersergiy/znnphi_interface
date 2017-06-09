from common import round_to_simd

class Tensor:
    def __init__(self, dim):
        self.dim  = dim
        self.size = 1

        self.size *= dim[0]
        self.size *= round_to_simd(dim[1])
        self.size *= round_to_simd(dim[2])
        self.size *= dim[3]
        self.size *= dim[4]
