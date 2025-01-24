import jax
import jax.numpy as jnp

from enum import IntEnum
import seaborn as sns

@jax.jit
def compute_NLL(preds, responses):
    eps = 1e-7
    preds_clipped = jnp.clip(preds, eps, 1 - eps)
    return -jnp.sum(
        responses * jnp.log(preds_clipped)
        + (1 - responses) * jnp.log(1 - preds_clipped)
    )

fig_dir = "../figures/outputs/"

colors = sns.color_palette("colorblind")
palette = {"refer to in-group": colors[0], "refer to mixed": colors[2], "social": colors[1]}

class ModelType(IntEnum):
    base = 0
    social = 1