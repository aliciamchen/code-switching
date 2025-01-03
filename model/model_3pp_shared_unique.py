# type: ignore
from memo import memo
import jax
from jax import lax
import jax.numpy as jnp

from enums import *


class Choices(IntEnum):
    Shared = 0
    GroupSpecific = 1


@jax.jit
def audience_wpp(audience_condition, audience):
    # for the "either group" condition, return 1 regardless of audience
    # for the "one group" condition, return 1 for the ingroup and 0 for the outgroup
    return jnp.array([1, audience])[audience_condition]


@jax.jit
def ref_info(audience, utterance):
    ingroup_info = jnp.array([1, 1])  # [shared, group-specific]
    outgroup_info = jnp.array([1, 0]) 

    info = lax.cond(
        audience == Audiences.Ingroup,
        lambda _: ingroup_info,
        lambda _: outgroup_info,
        operand=None,
    )

    return info[utterance]


@jax.jit
def social_info(utterance):
    return jnp.array([0, 1])[utterance]


@jax.jit
def cost(utterance):
    return jnp.array([1, 1])[utterance]


@memo
def speaker[
    utterance: Choices, audience: Audiences
](
    audience_condition: AudienceConditions,
    ttype,
    alpha,
    w_r,
    w_s,
    w_c,
):
    cast: [speaker]
    speaker: chooses(
        audience in Audiences, wpp=audience_wpp(audience_condition, audience)
    )
    speaker: chooses(
        utterance in Choices,
        wpp=exp(
            alpha
            * (
                w_r * ref_info(audience, utterance)
                + w_s * social_info(utterance)
                - w_c * cost(utterance)
            )
        ),
    )
    return Pr[speaker.utterance == utterance]
