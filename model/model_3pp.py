from memo import memo
import jax
import jax.numpy as jnp
from enum import IntEnum


class U(IntEnum):
    SHORT_SHARED_A = 1 << 0
    LONG_RED_A = 1 << 1
    LONG_BLUE_A = 1 << 2
    SHORT_SHARED_B = 1 << 3
    LONG_RED_B = 1 << 4
    LONG_BLUE_B = 1 << 5
    SHORT_SHARED_C = 1 << 6
    LONG_RED_C = 1 << 7
    LONG_BLUE_C = 1 << 8
    SHORT_RED_D = 1 << 9
    SHORT_BLUE_D = 1 << 10
    LONG_RED_D = 1 << 11
    LONG_BLUE_D = 1 << 12
    SHORT_RED_E = 1 << 13
    SHORT_BLUE_E = 1 << 14
    LONG_RED_E = 1 << 15
    LONG_BLUE_E = 1 << 16
    SHORT_RED_F = 1 << 17
    SHORT_BLUE_F = 1 << 18
    LONG_RED_F = 1 << 19
    LONG_BLUE_F = 1 << 20


class O(IntEnum):
    TANGRAM_A = U.SHORT_SHARED_A | U.LONG_RED_A | U.LONG_BLUE_A
    TANGRAM_B = U.SHORT_SHARED_B | U.LONG_RED_B | U.LONG_BLUE_B
    TANGRAM_C = U.SHORT_SHARED_C | U.LONG_RED_C | U.LONG_BLUE_C
    TANGRAM_D = U.SHORT_RED_D | U.SHORT_BLUE_D | U.LONG_RED_D | U.LONG_BLUE_D
    TANGRAM_E = U.SHORT_RED_E | U.SHORT_BLUE_E | U.LONG_RED_E | U.LONG_BLUE_E
    TANGRAM_F = U.SHORT_RED_F | U.SHORT_BLUE_F | U.LONG_RED_F | U.LONG_BLUE_F


class Z_s(IntEnum):
    RED = (
        U.SHORT_SHARED_A
        | U.LONG_RED_A
        | U.SHORT_SHARED_B
        | U.LONG_RED_B
        | U.SHORT_SHARED_C
        | U.LONG_RED_C
        | U.SHORT_RED_D
        | U.LONG_RED_D
        | U.SHORT_RED_E
        | U.LONG_RED_E
        | U.SHORT_RED_F
        | U.LONG_RED_F
    )
    BLUE = (
        U.SHORT_SHARED_A
        | U.LONG_BLUE_A
        | U.SHORT_SHARED_B
        | U.LONG_BLUE_B
        | U.SHORT_SHARED_C
        | U.LONG_BLUE_C
        | U.SHORT_BLUE_D
        | U.LONG_BLUE_D
        | U.SHORT_BLUE_E
        | U.LONG_BLUE_E
        | U.SHORT_BLUE_F
        | U.LONG_BLUE_F
    )

# Define costs in a dense array
COST_MAPPING = jnp.array([1 if "SHORT" in u.name else 3 for u in U])

@jax.jit
def cost(utterance: U):
    # Compute the dense index using log2 of the enum value
    dense_index = jnp.log2(utterance).astype(int)
    return COST_MAPPING[dense_index]

@jax.jit
def is_consistent(*args):
    result = args[0]
    for arg in args[1:]:
        result &= arg
    return result != 0


@memo
def L0_o[u: U, o: O, z_s: Z_s]():
    cast: [speaker, listener]
    listener: thinks[
        speaker : given(o in O, wpp=1),
        speaker : given(z_s in Z_s, wpp=1),
        speaker : chooses(u in U, wpp=is_consistent(u, o, z_s)),
    ]
    listener: observes[speaker.u] is u
    listener: chooses(o in O, wpp=Pr[speaker.o == o])
    listener: chooses(z_s in Z_s, wpp=Pr[speaker.z_s == z_s])
    return Pr[listener.o == o]


@memo
def L0_z[u: U, o: O, z_s: Z_s]():
    cast: [speaker, listener]
    listener: thinks[
        speaker : given(o in O, wpp=1),
        speaker : given(z_s in Z_s, wpp=1),
        speaker : chooses(u in U, wpp=is_consistent(u, o, z_s)),
    ]
    listener: observes[speaker.u] is u
    listener: chooses(o in O, wpp=Pr[speaker.o == o])
    listener: chooses(z_s in Z_s, wpp=Pr[speaker.z_s == z_s])
    return Pr[listener.z_s == z_s]


@memo
def S[u: U, o: O, z_s: Z_s](alpha):
    cast: [speaker, listener]
    speaker: knows(o)
    speaker: knows(z_s)
    speaker: chooses(u in U, wpp=exp(alpha * (L0_o[u, o, z_s]() + L0_z[u, o, z_s]() - cost(u))))
    return Pr[speaker.u == u]