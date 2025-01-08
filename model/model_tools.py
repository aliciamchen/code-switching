import jax
import jax.numpy as jnp
from jax import lax

from enums import *
import utils
import model_3pp_earlier_later
import model_3pp_shared_unique


@jax.jit
def get_model_preds(alpha, w_r, w_s, w_c, expt_type):
    """
    expt_type = 0 -> model_3pp_earlier_later
    expt_type = 1 -> model_3pp_shared_unique

    Get the model predictions for all conditions and tangram types
    Output: 
    expt_type = ExptType.EarlierLater -> 3x2 (condition, tangram type) array of probability of choosing the later utterance
    expt_type = ExptType.SharedUnique ->  3x2 (condition, tangram type) array array of probability of choosing the group-specific label (note that its the same across tangram types here)
    """
    def speaker_func(aud_cond, ttype, alpha, w_r, w_s, w_c, expt_type):
        return lax.cond(
            expt_type == ExptTypes.EarlierLater,
            lambda: model_3pp_earlier_later.speaker(aud_cond, ttype, alpha, w_r, w_s, w_c),
            lambda: model_3pp_shared_unique.speaker(aud_cond, ttype, alpha, w_r, w_s, w_c)
        )

    conditions_vals = jnp.array(
        [Conditions.ReferEither, Conditions.ReferOne, Conditions.SocialOne]
    )
    tangrams_vals = jnp.array([TangramTypes.Shared, TangramTypes.GroupSpecific])

    def single_pred(cond, ttype):
        return lax.cond(
            cond == Conditions.ReferEither,
            lambda _: speaker_func(AudienceConditions.EitherGroup, ttype, alpha, w_r, 0, w_c, expt_type),
            lambda _: lax.cond(
                cond == Conditions.ReferOne,
                lambda __: speaker_func(AudienceConditions.OneGroup, ttype, alpha, w_r, 0, w_c, expt_type),
                lambda __: speaker_func(AudienceConditions.OneGroup, ttype, alpha, w_r, w_s, w_c, expt_type),
                operand=None,
            ),
            operand=None,
        )[1, 0]  # probability of choosing the later utterance or the group-specific label

    # Vectorize over tangrams first, then over conditions
    vmap_tangrams = jax.vmap(
        lambda c: jax.vmap(lambda t: single_pred(c, t))(tangrams_vals)
    )
    return vmap_tangrams(conditions_vals)


# @jax.jit
def format_model_preds(model_preds, data_organized, tangram_info, expt_type):
    """format model predictions to match the data_organized format, which is a dict with keys (tangram_set, counterbalance)
    also duplicates the predictions by the number of participants
    needs tangram_info, which is information about what tangrams belong to which set and counterbalance etc.
    also needs data_organized (output of functions in data_tools)
    """
    model_organized = {}

    if expt_type == ExptTypes.EarlierLater:
        for key, mtx in data_organized.items():
            tangram_set, counterbalance = key
            preds = jnp.zeros((12, 2, 2, len(Conditions)))
            available = tangram_info[
                (tangram_info["tangram_set"] == tangram_set)
                & (tangram_info["counterbalance"] == counterbalance)
            ]
            for _, row in available.iterrows():
                for condition in Conditions:
                    preds = preds.at[Tangram[row["tangram"]], TangramTypes[row["tangram_type"]], :, condition].set(
                        model_preds[condition, TangramTypes[row["tangram_type"]]]
                    )

            # repeat by number of participants
            model_organized[key] = jnp.repeat(
                preds[:, :, :, :, jnp.newaxis], mtx.shape[-1], axis=-1
            )
    elif expt_type == ExptTypes.SharedUnique: 
        for key, mtx in data_organized.items(): 
            tangram_set, counterbalance = key
            preds = jnp.zeros((12, 12, 2, len(Conditions)))
            available = tangram_info[
                (tangram_info["tangram_set"] == tangram_set)
                & (tangram_info["counterbalance"] == counterbalance)
            ]
            for _, row in available.iterrows():
                for condition in Conditions:
                    preds = preds.at[Tangram[row["shared.tangram"]], Tangram[row["unique.tangram"]], :, condition].set(
                        model_preds[condition, 0] # both columns are the same because tangram type here doesn't matter
                    )

            # repeat by number of participants
            model_organized[key] = jnp.repeat(
                preds[:, :, :, :, jnp.newaxis], mtx.shape[-1], axis=-1
            )


    return model_organized


def fit_params_overall(data_organized, tangram_info, 
                    params_list, expt_type):
    """Data organized is a dict with keys (tangram_set, counterbalance) and values 12 x 2 x 3 x n_participants
    """
    model_slices = utils.get_surviving_slices(
        data_organized, expt_type
    )  # precompute slices of unused tangrams, to avoid jax issues
    data_slices = utils.get_surviving_slices(data_organized, expt_type)
    data_all = utils.make_stacked_mtx(data_organized, data_slices)

    def single_nll(params):
        alpha, w_r, w_s, w_c = params
        preds = get_model_preds(alpha, w_r, w_s, w_c, expt_type)
        model_organized = format_model_preds(preds, data_organized, tangram_info, expt_type)
        model_all = utils.make_stacked_mtx(model_organized, model_slices)

        return compute_nll(data_all, model_all)

    nll_values = jax.vmap(single_nll)(params_list)

    # Find best NLL
    best_idx = jnp.argmin(nll_values)
    assert best_idx < len(nll_values)
    best_nll = nll_values[best_idx]
    best_params = params_list[best_idx]
    return best_params, best_nll, nll_values

def fit_params_participant(data_organized, tangram_info, params_list, expt_type): 
    """
    Fit parameters for each participant separately
    """
    data_all = utils.make_stacked_mtx(data_organized)

    def single_nlls(params):
        """Returns NLL for each participant
        Output: n_participants array of NLLs
        """
        alpha, w_r, w_s, w_c = params
        preds = get_model_preds(alpha, w_r, w_s, w_c, expt_type)
        model_organized = format_model_preds(preds, data_organized, tangram_info, expt_type)
        model_all = utils.make_stacked_mtx(model_organized)

        # the last dimension of data_all and model_all is the participant dimension
        assert data_all.shape[-1] == model_all.shape[-1]
        total_participants = data_all.shape[-1]
        nlls = jax.vmap(lambda i: compute_nll(data_all[..., i], model_all[..., i]))(jnp.arange(total_participants))
        return nlls

    nll_values = jax.vmap(single_nlls)(params_list)  # shape: (n_params, n_participants)
    # TODO: verify the shape of nll_values

    # for each participant, find the best NLL and parameters
    best_params = []
    best_nlls = []
    for i in range(nll_values.shape[-1]): 
        best_idx = jnp.argmin(nll_values[..., i])
        best_nll = nll_values[best_idx, i]
        best_params.append(params_list[best_idx])
        best_nlls.append(best_nll)
    return best_params, best_nlls, nll_values


@jax.jit
def compute_nll(data, model):
    assert data.shape == model.shape
    nll = -jnp.sum(
        data * jnp.log(model + 1e-7) + (1 - data) * jnp.log(1 - model + 1e-7)
    )
    return nll
