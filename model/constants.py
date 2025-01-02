import jax.numpy as jnp
# Make grid of parameter values
alphas = jnp.arange(0, 5, 0.1)
w_rs = jnp.arange(0, 5, 0.1)
w_ss = jnp.arange(0, 5, 0.1)
w_cs = jnp.arange(0, 5, 0.1)

# make grid of all values
param_grid = jnp.meshgrid(alphas, w_rs, w_ss, w_cs)

# Flatten the parameter grid
all_alphas = param_grid[0].ravel()
all_wrs    = param_grid[1].ravel()
all_wss    = param_grid[2].ravel()
all_wcs    = param_grid[3].ravel()

params_list = jnp.stack([all_alphas, all_wrs, all_wss, all_wcs], axis=1)