import os
import json
import timeit
import pickle
import itertools
from abc import ABC, abstractmethod
from pathlib import Path
from multiprocessing import cpu_count

import numpy as np


def get_from_config():
    """Get benchmarks configuration from the config.json file"""
    current_path = Path(__file__).resolve().parent

    config_path = current_path / "config.json"
    with open(config_path, "r") as config_file:
        config_file = "".join(line for line in config_file if line and "//" not in line)
        config = json.loads(config_file)

    profile = os.getenv("SKLBENCH_PROFILE", config["profile"])

    n_jobs_vals_env = os.getenv("SKLBENCH_NJOBS")
    if n_jobs_vals_env:
        n_jobs_vals = eval(n_jobs_vals_env)
    else:
        n_jobs_vals = config["n_jobs_vals"]
    if not n_jobs_vals:
        n_jobs_vals = list(range(1, 1 + cpu_count()))

    cache_path = current_path / "cache"
    cache_path.mkdir(exist_ok=True)
    (cache_path / "estimators").mkdir(exist_ok=True)
    (cache_path / "tmp").mkdir(exist_ok=True)

    save_estimators = os.getenv("SKLBENCH_SAVE_ESTIMATORS", config["save_estimators"])
    save_dir = os.getenv("ASV_COMMIT", "new")[:8]

    if save_estimators:
        (cache_path / "estimators" / save_dir).mkdir(exist_ok=True)

    base_commit = os.getenv("SKLBENCH_BASE_COMMIT", config["base_commit"])

    bench_predict = os.getenv("SKLBENCH_PREDICT", config["bench_predict"])
    bench_transform = os.getenv("SKLBENCH_TRANSFORM", config["bench_transform"])

    return (
        profile,
        n_jobs_vals,
        save_estimators,
        save_dir,
        base_commit,
        bench_predict,
        bench_transform,
    )


def get_estimator_path(benchmark, directory, params, save=False):
    """Get path of pickled fitted estimator"""
    path = Path(__file__).resolve().parent / "cache"
    path = (path / "estimators" / directory) if save else (path / "tmp")

    filename = (
        benchmark.__class__.__name__
        + "_estimator_"
        + "_".join(list(map(str, params)))
        + ".pkl"
    )

    return path / filename


def clear_tmp():
    """Clean the tmp directory"""
    path = Path(__file__).resolve().parent / "cache" / "tmp"
    for child in path.iterdir():
        child.unlink()


class Benchmark(ABC):
    """Abstract base class for all the benchmarks"""

    timer = timeit.default_timer  # wall time
    processes = 1
    timeout = 500

    (
        profile,
        n_jobs_vals,
        save_estimators,
        save_dir,
        base_commit,
        bench_predict,
        bench_transform,
    ) = get_from_config()

    if profile == "fast":
        warmup_time = 0
        repeat = 1
        number = 1
        min_run_count = 1
        data_size = "small"
    elif profile == "regular":
        warmup_time = 1
        repeat = (3, 100, 30)
        data_size = "small"
    elif profile == "large_scale":
        warmup_time = 1
        repeat = 3
        number = 1
        data_size = "large"

    @property
    @abstractmethod
    def params(self):
        pass
