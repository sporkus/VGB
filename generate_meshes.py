#!/usr/bin/env python3
import json
import os
import sys
from pprint import pprint
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import json


def import_measurements(fp, sensor="frame_temp"):
    with open(fp) as f:
        data = json.load(f)["measurements"]

    temps = [x[sensor] for x in data.values()]
    meshs = [x["mesh"] for x in data.values()]
    return temps, meshs


def make_temp_matrix(temp: tuple, shape: tuple) -> list:
    """Generate interpolated temperatures for in a cartesian grid.
    Since sampling was done in a zig-zag path, the array
    on even rows needs to be flipped

    Args:
        temp (tuple): start,end temp
        shape (tuple): shape of the probed matrix

    Returns:
        list: A 2D array of temperatures that cooreponds to the probed points
    """
    temps = np.linspace(*temp, num=np.product(shape)).reshape(shape)

    ordered_temps = []
    for j, row in enumerate(temps):
        if j % 2:
            row = np.flip(row)
        ordered_temps.append(row.tolist())

    return np.array(ordered_temps)


def get_mesh_params(mesh):
    return mesh["profiles"]["default"]["mesh_params"]


def sample_fit(temps: list, probed_matrices: list) -> np.ndarray:
    """Fitting each probed point indepdently

    Args:
        temps (list): Array of (start temp, end temp)
        probed_matrices (list): Array of probed_matrices that pairs with the temperatures

    Returns:
        list: np.polyfit() coeffs in a cartesian array that pairs with probed_matrices
    """
    try:
        assert len(temps) == len(probed_matrices)
    except:
        raise AssertionError("Did not have matching number of temperature/mesh samples")

    n_sample = len(temps)
    mesh_shape = np.array(probed_matrices[0]).shape
    temp_matrices = [make_temp_matrix(x, mesh_shape) for x in temps]

    # Convert to array of 1d array and transpose
    # after transpose, each row represents one coordinate
    # data's shape should be (# of probe points, # of samples)
    # ts: start..end temp; zs: cold..hot probed height
    shape1d = np.product(mesh_shape)
    ts = np.array(temp_matrices).reshape(n_sample, shape1d).transpose()
    zs = np.array(probed_matrices).reshape(n_sample, shape1d).transpose()

    coeffs = []
    for i in range(shape1d):
        coeffs.append(np.polyfit(ts[i], zs[i], deg=1))
    coeffs_2d = np.array(coeffs).reshape((mesh_shape[0], mesh_shape[1], 2))
    return coeffs_2d


def predict_mesh(temp: float, coeff_matrix: np.ndarray) -> np.ndarray:
    """Predicted mesh height at any temperature

    Args:
        temp (float): temperature for prediction
        coeff_matrix (np.ndarray): coeffs of the height/temp model in cartesian grid
        in the shape of (x probes, y probes, # of coefficients)

    Returns:
        np.ndarray: predicted mesh
    """

    # Shape of coeff matrix is (count_x, count_y, # of coefficients)
    n_x, n_y, n_coeff = coeff_matrix.shape
    coeff_arr = coeff_matrix.reshape(n_x * n_y, n_coeff)

    predict1d = []
    for coeff in coeff_arr:
        p = np.poly1d(coeff)
        predict1d.append(p(temp))

    return np.array(predict1d).reshape(n_x, n_y)


def make_meshgrid(mesh_params):
    pts = {}
    for axis in ("x", "y"):
        min = mesh_params[f"min_{axis}"]
        max = mesh_params[f"max_{axis}"]
        num = mesh_params[f"{axis}_count"]
        pts[axis] = np.linspace(min, max, num).round()

    return np.meshgrid(pts["x"], pts["y"])


def generate_config(temps, mesh_params, coeffs):
    config = []
    param_lines = [f"{k} = {v}" for k, v in mesh_params.items()]

    # Temperature range to predict (1C above/below measured range)
    temps = np.array(temps)
    for t in np.arange(temps.min() - 1, temps.max() + 1, step=0.1):
        predicted = predict_mesh(t, coeffs).round(6).astype(str)

        config.append(f"[bed_mesh linear_reg_{t:.2f}]")
        config.append("version = 1")
        config.append("points =")
        for row in predicted:
            config.append("    " + ", ".join(row))
        config.extend(param_lines)
        config.append("")

    return ["\n#*# " + line for line in config]


def main(data_fp):
    temps, meshs = import_measurements(data_fp)
    mesh_params = get_mesh_params(meshs[0])
    probed_matrices = [x["probed_matrix"] for x in meshs]
    coeffs = sample_fit(temps, probed_matrices)
    xs, ys = make_meshgrid(mesh_params)

    cfg = generate_config(temps, mesh_params, coeffs)
    with open("generated_meshs.cfg", "w") as f:
        f.writelines(cfg)


if __name__ == "__main__":
    args = sys.argv[1:]
    main(args[0])
