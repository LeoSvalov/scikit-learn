import numpy as np
import pytest
from sklearn.neighbors._navigable_small_world_graph import NSWGraph
from joblib import Parallel
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_array
from numpy.testing import assert_array_almost_equal

rng = np.random.RandomState(10)

# def test_array_object_type():
#     """Check that we do not accept object dtype array."""
    # X = np.array([(1, 2, 3), (2, 5), (5, 5, 1, 2)], dtype=object)
    # with pytest.raises(ValueError, match="setting an array element with a sequence"):
    #     NSWGraph()

def brute_force_neighbors(X, Y, k, metric, **kwargs):
    from sklearn.metrics import DistanceMetric

    X, Y = check_array(X), check_array(Y)
    D = DistanceMetric.get_metric(metric, **kwargs).pairwise(Y, X)
    ind = np.argsort(D, axis=1)[:, :k]
    dist = D[np.arange(Y.shape[0])[:, None], ind]
    return dist, ind


def test_query():
    rng = check_random_state(0)
    X = rng.random_sample((40, 16))
    g = NSWGraph(n_neighbors=3)
    g.build_navigable_graph(X)
    hops, ind1 = g.query(X[:3])
    dist2, ind2 = brute_force_neighbors(X, X[:3], k=3, metric="euclidean")
    # assert_array_almost_equal(dist1, dist2)
    assert_array_almost_equal(ind1, ind2)
