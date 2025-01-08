"""General tools"""

import numpy as np
import jax
import jax.numpy as jnp

from enums import *


def squeeze_unused_tangrams(organized_dict):
    """
    Input: dict with keys (tangram_set, counterbalance) and values 12 x 2 x 3 x n_participants
    For each tangram_set and counterbalance, get rid of unused tangrams along 0th dim
    """
    squeezed_dict = {}
    for key, mtx in organized_dict.items():
        squeezed_mtx = mtx[~np.all(mtx == 0, axis=tuple(range(1, mtx.ndim)))]
        squeezed_dict[key] = squeezed_mtx
    return squeezed_dict


def get_surviving_slices(organized_dict, expt_type):
    """
    Input: dict (either for data or model) with keys (tangram_set, counterbalance) and values mtx
    Output: list of tuples with (tangram_set, counterbalance) keys and surviving indices values
    """
    # TODO: figure out a way to slice out first TWO dimensions, for both experiments
    surviving_slices = []
    for key, mtx in organized_dict.items():
        mask = ~jnp.all(mtx == 0, axis=tuple(range(1, mtx.ndim)))
        surviving_indices = jnp.where(mask)[0]
        surviving_slices.append((key, surviving_indices))
        # TODO: be able to take in first AND second indices/ dims
    return surviving_slices


def make_stacked_mtx(organized_dict, surviving_slices=None):
    """
    Filter unused tangrams and stack remaining matrices
    TODO: vectorize this
    """
    masked_arrays = []
    if surviving_slices:
        for key, surviving_indices in surviving_slices:
            mtx = organized_dict[key]
            masked_arrays.append(mtx[surviving_indices])  # (k, 2, 3, Nx)
    else:
        for mtx in organized_dict.values():
            masked_arrays.append(mtx)
    return jnp.concatenate(masked_arrays, axis=-1)


def get_params_list(alphas, w_rs, w_ss, w_cs):
    param_grid = jnp.meshgrid(alphas, w_rs, w_ss, w_cs, indexing="ij")

    # Flatten the parameter grid
    all_alphas = param_grid[0].ravel()
    all_wrs = param_grid[1].ravel()
    all_wss = param_grid[2].ravel()
    all_wcs = param_grid[3].ravel()

    # Stack the flattened parameters into a single array
    params_list = jnp.stack([all_alphas, all_wrs, all_wss, all_wcs], axis=1)
    return params_list
