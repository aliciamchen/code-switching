"""General tools"""
import numpy as np
import jax
import jax.numpy as jnp


def squeeze_unused_tangrams(organized_dict):
    """
    Input: dict with keys (tangram_set, counterbalance) and values 12 x 2 x 3 x n_participants
    For each tangram_set and counterbalance, get rid of unused tangrams along 0th dim
    """
    squeezed_dict = {}
    for key, mtx in organized_dict.items():
        squeezed_mtx = mtx[~np.all(mtx == 0, axis=(1, 2, 3))]
        squeezed_dict[key] = squeezed_mtx
    return squeezed_dict

def get_surviving_slices(organized_dict):
    """
    Input: dict (either for data or model) with keys (tangram_set, counterbalance) and values 12 x 2 x 3 x n_participants
    Output: list of tuples with (tangram_set, counterbalance) keys and surviving indices values
    """
    surviving_slices = []
    for key, mtx in organized_dict.items():
        mask = ~jnp.all(mtx == 0, axis=(1, 2, 3))
        surviving_indices = jnp.where(mask)[0]
        surviving_slices.append((key, surviving_indices))
    return surviving_slices

def make_stacked_mtx(organized_dict, surviving_slices): 
    """
    Filter unused tangrams and stack remaining matrices
    TODO: vectorize this
    """
    masked_arrays = []
    for key, surviving_indices in surviving_slices:
        mtx = organized_dict[key]
        masked_arrays.append(mtx[surviving_indices])  # (k, 2, 3, Nx)
    return jnp.concatenate(masked_arrays, axis=-1)




