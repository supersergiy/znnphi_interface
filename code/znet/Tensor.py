class Tensor:
    def __init__(self, dim):
        self.dim  = dim
        self.size = 1
        for d in self.dim:
            self.size *= d
        self.size = int(self.size)
