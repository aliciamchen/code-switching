import jax
import jax.numpy as jnp
from jax import lax

from enums import *
import utils
import model_3pp_earlier_later


@jax.jit
def get_model_preds(alpha, w_r, w_s, w_c):
    """
    Get the model predictions for all conditions and tangram types
    Output: 3x2 (condition, tangram type) array of probability of choosing the later utterance
    """
    conditions_vals = jnp.array(
        [Conditions.ReferEither, Conditions.ReferOne, Conditions.SocialOne]
    )
    tangrams_vals = jnp.array([TangramTypes.Shared, TangramTypes.GroupSpecific])

    def single_pred(cond, ttype):
        return lax.cond(
            cond == Conditions.ReferEither,
            lambda _: model_3pp_earlier_later.speaker(
                AudienceConditions.EitherGroup, ttype, alpha, w_r, 0, w_c
            ),
            lambda _: lax.cond(
                cond == Conditions.ReferOne,
                lambda __: model_3pp_earlier_later.speaker(
                    AudienceConditions.OneGroup, ttype, alpha, w_r, 0, w_c
                ),
                lambda __: model_3pp_earlier_later.speaker(
                    AudienceConditions.OneGroup, ttype, alpha, w_r, w_s, w_c
                ),
                operand=None,
            ),
            operand=None,
        )[EarlierLaterChoices.Later, 0]

    # Vectorize over tangrams first, then over conditions
    vmap_tangrams = jax.vmap(
        lambda c: jax.vmap(lambda t: single_pred(c, t))(tangrams_vals)
    )
    return vmap_tangrams(conditions_vals)


# @jax.jit
def format_model_preds(model_preds, data_organized, tangram_info):
    """format model predictions to match the data_organized format, which is a dict with keys (tangram_set, counterbalance)
    also duplicates the predictions by the number of participants
    needs tangram_info, which is information about what tangrams belong to which set and counterbalance etc.
    also needs data_organized (output of functions in data_tools)
    """
    model_organized = {}
    for key, mtx in data_organized.items():
        tangram_set, counterbalance = key
        preds = jnp.zeros((12, 2, len(Conditions)))
        available = tangram_info[
            (tangram_info["tangram_set"] == tangram_set)
            & (tangram_info["counterbalance"] == counterbalance)
        ]
        for _, row in available.iterrows():
            for condition in Conditions:
                preds = preds.at[Tangram[row["tangram"]], :, condition].set(
                    model_preds[condition, TangramTypes[row["tangram_type"]]]
                )

        # repeat by number of participants
        model_organized[key] = jnp.repeat(
            preds[:, :, :, jnp.newaxis], mtx.shape[-1], axis=-1
        )
    return model_organized


def grid_search_nll(data_organized, tangram_info, 
                    params_list):
    """Data organized is a dict with keys (tangram_set, counterbalance) and values 12 x 2 x 3 x n_participants
    Model type is either "social" or "no_social", where for the latter w_s is fixed to 0
    """

    model_slices = utils.get_surviving_slices(
        data_organized
    )  # precompute slices of unused tangrams, to avoid jax issues
    data_slices = utils.get_surviving_slices(data_organized)
    data_all = utils.make_stacked_mtx(data_organized, data_slices)

    def single_nll(params):
        alpha, w_r, w_s, w_c = params
        preds = get_model_preds(alpha, w_r, w_s, w_c)
        model_organized = format_model_preds(preds, data_organized, tangram_info)
        model_all = utils.make_stacked_mtx(model_organized, model_slices)

        return compute_nll(data_all, model_all)

    nll_values = jax.vmap(single_nll)(params_list)

    # Find best NLL
    best_idx = jnp.argmin(nll_values)
    assert best_idx < len(nll_values)
    best_nll = nll_values[best_idx]
    best_params = params_list[best_idx]
    return best_params, best_nll, nll_values


@jax.jit
def compute_nll(data, model):
    assert data.shape == model.shape
    nll = -jnp.sum(
        data * jnp.log(model + 1e-7) + (1 - data) * jnp.log(1 - model + 1e-7)
    )
    return nll
