from sklearn.neighbors import NSWGraph
from .common import Benchmark
import numpy as np


class NSWGSuite(Benchmark):
    def setup(self):
        self.data = data = np.array([[1, 2, 2, 1, 1, 2],
               [7, 8, 9, 10, 11, 12],
               [1, 4, 1, 2, 1, 3],
               [9, 11, 7, 6, 2, 1],
               [7, 2, 1, 2, 3, 1],
               [4, 6, 2, 5, 1, 4],
               [2, 5, 7, 11, 1, 8],
               [4, 1, 1, 2, 6, 3],
               [3, 10, 2, 6, 1, 1],
               [1, 2, 1, 2, 3, 2],
               [8, 12, 1, 6, 10, 2],
               [12, 11, 8, 10, 11, 12] ])
        self.k = 3
        self.query = np.array([[2, 1, 2, 1, 3, 1]])

    def time_build_index(self):
        reg = None
        attempts = 2
        guard_hops = 100
        quantize = False
        levels = 20
        self.index = NSWGraph(n_nodes=len(self.data), dimensions=len(self.data[0]), reg=reg, guard_hops=guard_hops)
        self.index.build_navigable_graph(self.data, attempts=attempts, quantize=quantize, quantization_levels=levels)

    def time_predict(self):
        guard_hops = 100
        approximated_neigbours = self.index.knnQueryBatch(self.query, top=self.k, guard_hops=guard_hops)

