# type: ignore
from memo import memo
import jax
import jax.numpy as jnp

from enums import *


class Choices(IntEnum):
    Earlier = 0
    Later = 1


# Referential info
# Earlier utterance: 1 for each audience member
# Later utterance: 1 for each in-group member, 0 for each out-group member

# Social info
# Earlier utterance: 0 for each audience member
# Later utterance: 1 for each in-group member, 0 for each out-group member

# Cost
# Earlier utterance: 1
# Later utterance: 0


@jax.jit
def ref_info(n_ingroup, n_outgroup, utterance):
    earlier_utility = n_ingroup + n_outgroup
    later_utility = n_ingroup
    return jnp.array([earlier_utility, later_utility])[utterance]


@jax.jit
def social_info(n_ingroup, n_outgroup, utterance):
    earlier_utility = 0
    later_utility = n_ingroup
    return jnp.array([earlier_utility, later_utility])[utterance]


@jax.jit
def cost(utterance):
    return jnp.array([1, 0])[utterance]


@memo
def speaker[utterance: Choices](n_ingroup, n_outgroup, alpha, w_r, w_s, w_c):
    cast: [speaker]
    speaker: chooses(
        utterance in Choices,
        wpp=exp(
            alpha
            * (
                w_r * ref_info(n_ingroup, n_outgroup, utterance)
                + w_s * social_info(n_ingroup, n_outgroup, utterance)
                - w_c * cost(utterance)
            )
        ),
    )
    return Pr[speaker.utterance == utterance]
