# distutils: language = c++
#!python
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

cimport numpy as np
np.import_array()
from libcpp.vector cimport vector
from libcpp.set cimport set as set_c
from libcpp.pair cimport pair as pair
from libc.math cimport pow
from libcpp.queue cimport priority_queue
from libc.stdlib cimport rand
import itertools
import numpy as np

cdef class NSWGraph:
    def __init__(self, ITYPE_t n_neighbors=1,
                       ITYPE_t regularity=16,
                       ITYPE_t guard_hops=100,
                       ITYPE_t attempts=2,
                       BTYPE_t quantize=False,
                       ITYPE_t quantization_levels=20):
        self.n_neighbors = n_neighbors
        self.regularity = regularity
        self.guard_hops = guard_hops
        self.attempts = attempts
        self.quantize = quantize,
        self.quantization_levels = quantization_levels

# todo: add values validation IN METHODS, not in init


    cdef priority_queue[pair[DTYPE_t, ITYPE_t]] delete_duplicate(self, priority_queue[pair[DTYPE_t, ITYPE_t]] queue) nogil:
        cdef priority_queue[pair[DTYPE_t, ITYPE_t]] new_que
        cdef set_c[ITYPE_t] tmp_set
        new_que.push(queue.top())
        tmp_set.insert(queue.top().second)
        queue.pop()
        while queue.size() != 0:
            if tmp_set.find(queue.top().second) == tmp_set.end():
                tmp_set.insert(queue.top().second)
                new_que.push(queue.top())
            queue.pop()
        return new_que

    cdef DTYPE_t eucl_dist(self, vector[DTYPE_t] v1, vector[DTYPE_t] v2) nogil:
        cdef ITYPE_t i = 0
        cdef DTYPE_t res = 0
        if self.quantize:
            for i in range(v1.size()):
            # for i in prange(v1.size(), nogil=True):
                res += self.lookup_table[int(v2[i])][int(v1[i])]
        else:
            for i in range(v1.size()):
            # for i in prange(v1.size(), nogil=True):
                res += pow(v1[i] - v2[i], 2)
        return res


    cdef void search_nsw_basic(self, vector[DTYPE_t] query,
                               set_c[ITYPE_t]* visitedSet,
                               priority_queue[pair[DTYPE_t, ITYPE_t]]* candidates,
                               priority_queue[pair[DTYPE_t, ITYPE_t]]* result,
                               ITYPE_t* res_hops,
                               ITYPE_t k) nogil:

        cdef ITYPE_t entry = rand() % self.nodes.size()
        cdef ITYPE_t hops = 0
        cdef DTYPE_t closest_dist = 0
        cdef ITYPE_t closest_id = 0
        cdef ITYPE_t e = 0
        cdef DTYPE_t d = 0
        cdef pair[DTYPE_t, ITYPE_t] tmp_pair

        d = self.eucl_dist(query, self.nodes[entry])
        tmp_pair.first = d * (-1)
        tmp_pair.second = entry

        if visitedSet[0].find(entry) == visitedSet[0].end():
            candidates[0].push(tmp_pair)
        tmp_pair.first = tmp_pair.first * (-1)
        result[0].push(tmp_pair)
        hops = 0

        while hops < self.guard_hops:
            hops += 1
            if candidates[0].size() == 0:
                break
            tmp_pair = candidates[0].top()
            candidates.pop()
            closest_dist = tmp_pair.first * (-1)
            closest_id = tmp_pair.second
            if result[0].size() >= k:
                while result[0].size() > k:
                    result[0].pop()

                if result[0].top().first < closest_dist:
                    break

            for e in self.neighbors[closest_id]:
                if visitedSet[0].find(e) == visitedSet[0].end():
                    d = self.eucl_dist(query, self.nodes[e])
                    visitedSet[0].insert(e)
                    tmp_pair.first = d
                    tmp_pair.second = e
                    result.push(tmp_pair)
                    tmp_pair.first = tmp_pair.first * (-1)
                    candidates.push(tmp_pair)
        res_hops[0] = hops


    cdef np.ndarray find_quantized_values(self, np.ndarray vector):
      result = []
      for i, data_value in enumerate(vector):
        result.append((np.abs(self.quantization_values - data_value)).argmin())
      return np.array(result)


    cdef pair[vector[ITYPE_t], ITYPE_t] _multi_search(self, vector[DTYPE_t] query, ITYPE_t k) nogil:
        cdef set_c[ITYPE_t] visitedSet
        cdef priority_queue[pair[DTYPE_t, ITYPE_t]] candidates
        cdef priority_queue[pair[DTYPE_t, ITYPE_t]] result
        cdef vector[ITYPE_t] res
        cdef ITYPE_t i
        cdef ITYPE_t hops
        cdef pair[DTYPE_t, ITYPE_t] j
        cdef ITYPE_t id

        for i in range(self.attempts):
            self.search_nsw_basic(query, &visitedSet, &candidates, &result, &hops, k)
            result = self.delete_duplicate(result)

        while result.size() > k:
            result.pop()
        while res.size() < k:
            el = result.top().second
            res.push_back(el)
            if not result.empty():
                result.pop()
            else:
                break
        return pair[vector[ITYPE_t], ITYPE_t](res, hops)

    cdef np.ndarray run_quantization(self, np.ndarray data):
        self.quantization_values = np.linspace(0.0, 1.0, num=self.quantization_levels)
        self.lookup_table = np.zeros(shape=(self.quantization_levels,self.quantization_levels))
        for v in itertools.combinations(enumerate(self.quantization_values), 2):
            i = v[0][0]
            j = v[1][0]
            self.lookup_table[i][j] = pow(np.abs(v[0][1]-v[1][1]),2)
            self.lookup_table[j][i] = pow(np.abs(v[1][1]-v[0][1]),2)
        quantized_data = []
        for i, vector in enumerate(data):
            quantized_data.append(self.find_quantized_values(vector))
        return np.array(quantized_data)

    cdef ITYPE_t _build_navigable_graph(self, vector[vector[DTYPE_t]] values) nogil:
        cdef vector[DTYPE_t] val
        cdef vector[ITYPE_t] closest
        cdef ITYPE_t c
        cdef ITYPE_t i
        cdef vector[ITYPE_t] res
        cdef set_c[ITYPE_t] tmp_set
        if values.size() != self.number_nodes:
            raise Exception("Number of nodes don't match")
        if values[0].size() != self.dimension:
            raise Exception("Dimension doesn't match")

        self.nodes.push_back(values[0])
        for i in range(self.number_nodes):
            self.neighbors.push_back(tmp_set)

        for i in range(1, self.number_nodes):
            val = values[i]
            closest.clear()
            closest = self._multi_search(val, k=self.regularity).first
            self.nodes.push_back(val)
            for c in closest:
                self.neighbors[i].insert(c)
                self.neighbors[c].insert(i)

    cdef vector[vector[DTYPE_t]] ndarray_to_vector_2(self, np.ndarray array):
        cdef vector[vector[DTYPE_t]] tmp_result
        cdef ITYPE_t i
        for i in range(len(array)):
            tmp_result.push_back((array[i]))
        return tmp_result

########################################################################################################################
    def get_params(self, deep=True):
        return {"n_neighbors": self.n_neighbors,
                "regularity": self.regularity,
                "guard_hops": self.guard_hops,
                "attempts": self.attempts,
                "quantize": self.quantize,
                "quantization_levels": self.quantization_levels}
    
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def search_nsw_basic_wrapped(self, np.ndarray query, ITYPE_t k=0):
        if k < 0:
            raise Exception("Incorrect number of desired neigbors")
        elif k == 0:
            k = self.n_neighbors

        cdef set_c[ITYPE_t] visitedSet
        cdef priority_queue[pair[DTYPE_t, ITYPE_t]] candidates
        cdef priority_queue[pair[DTYPE_t, ITYPE_t]] result
        cdef vector[ITYPE_t] res
        cdef ITYPE_t hops = 0
        self.search_nsw_basic(query, &visitedSet, &candidates, &result, &hops, k)
        result = self.delete_duplicate(result)
        while result.size() > k:
            result.pop()
        for i in range(result.size()):
            res.push_back(result.top().second)
            result.pop()
        return res, hops


    def build_navigable_graph(self, np.ndarray X):
        self.number_nodes = len(X)
        self.dimension = len(X[0])
        if self.quantize:
            normalized_values = X
            quantized_data = self.run_quantization(X)
            X = quantized_data
        cdef vector[vector[DTYPE_t]] tmp_result = self.ndarray_to_vector_2(X)
        self._build_navigable_graph(tmp_result)

    def fit(self, X, y):
        self.targets = y
        self.build_navigable_graph(X)


    def query(self, np.ndarray queries, ITYPE_t k=0):
        '''

        '''
        ind = []
        dist = []
        if k < 0:
            raise Exception("Incorrect number of desired neigbors.")
        elif k == 0:
            k = self.n_neighbors
        cdef pair[vector[ITYPE_t], ITYPE_t] res
        cdef vector[vector[DTYPE_t]] tmp
        for query in queries:
            if self.quantize:
                normalized_query = query
                query = self.find_quantized_values(normalized_query)
            query = np.array([query])
            tmp = self.ndarray_to_vector_2(query)
            res = self._multi_search(tmp[0], k)
            ind.append(res.first[::-1])
            dist.append(res.second)
        return np.array(dist), np.array(ind)


    def predict(self, X):
        hops, ind = self.query(X, k=1)
        result = np.array([self.targets[res[0]] for res in ind])
        return result
