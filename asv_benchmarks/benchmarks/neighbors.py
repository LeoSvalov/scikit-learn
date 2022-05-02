from sklearn.neighbors import KNeighborsClassifier, NSWGraph

from .common import Benchmark, Estimator, Predictor
from .datasets import _20newsgroups_lowdim_dataset
from .utils import make_gen_classif_scorers


class KNeighborsClassifierBenchmark(Predictor, Estimator, Benchmark):
    """
    Benchmarks for KNeighborsClassifier.
    """

    param_names = ["algorithm", "dimension", "n_jobs"]
    params = (["brute", "kd_tree", "ball_tree", "nswg_8", "nswg_16"],
              ["low", "high", "super-high", "mega-high"], Benchmark.n_jobs_vals)


    def setup_cache(self):
        super().setup_cache()

    def make_data(self, params):
        algorithm, dimension, n_jobs = params

        if Benchmark.data_size == "large":
            # n_components = 40 if dimension == "low" else 200
            if dimension == "low":
                n_components = 40
            elif dimension == "high":
                n_components = 200
            elif dimension == "super-high":
                n_components = 500
            else:
                n_components = 1000
        else:
            # n_components = 10 if dimension == "low" else 50
            if dimension == "low":
                n_components = 10
            elif dimension == "high":
                n_components = 50
            elif dimension == "super-high":
                n_components = 250
            else:
                n_components = 500

        data = _20newsgroups_lowdim_dataset(n_components=n_components)

        return data

    def make_estimator(self, params):
        algorithm, dimension, n_jobs = params

        if algorithm != "nswg_8" or algorithm != "nswg_16":
            estimator = KNeighborsClassifier(algorithm=algorithm, n_jobs=n_jobs)
        else:
            regularity = 16 if algorithm == "nswg_16" else 8
            estimator = NSWGraph(regularity)

        return estimator

    def make_scorers(self):
        make_gen_classif_scorers(self)