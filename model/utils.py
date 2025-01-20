import jax
import jax.numpy as jnp

@jax.jit
def compute_NLL(preds, responses):
    eps = 1e-7
    preds_clipped = jnp.clip(preds, eps, 1 - eps)
    return -jnp.sum(
        responses * jnp.log(preds_clipped)
        + (1 - responses) * jnp.log(1 - preds_clipped)
    )