# Modeling

## Sandbox 

- `2024-12_model-test-3pp-earlier-later.ipynb`: plug in various intuitive values for the utilities and see if predictions make sense
- `2024-12-23_fit-3pp-earlier-later.ipynb`: first try at model fitting (Dec 2024), fits a different set of weights per condition, has a bunch of assumptions about stuff, some hacky solutions and doesn't run all the way through
- `model_consistency_listener.py`: both the speaker and listener are aware of the entire lexicon, and informativity (both referential and social) is just based on whether a speaker's utterances are consistent with their group membership and the lexicon, so then based on this the listener can jointly infer speaker group membership and intended object


## Model fitting strategy for 3rd party experiments

- General strategy: all parameters without social utility term (i.e. freeze that weight to zero), then add back in social utility term only for the 'social goal' trials, refit all parameters, and see if additional social utility term makes the model do better
- First, do this using binary utilities directly corresponding to the experimental design (basically this means there is one set of predictions to compare each participant to because this doesn't consider the tangram sets + counterbalancing), and then try accounting for the between-trial + between-set differences using measured values from experiment(s), and see if that improves fit. Note: one thing that we can't directly measure is social value of utterances, so we can think about how to do this (maybe it's connected to the 'arbitrariness' of an utterance?)
- Later, we can also fit parameters to individual participants (to look at the distribution of fit parameters — do certain participants consider social goals more than others?)

## Binary utilites to put in model 

### Earlier-later experiment

see `model_3pp_earlier_later.py`

- Referential informativity
    - 1 for all earlier utterances regardless of group; 1 for later utterances to in-group; 0 for later utterances to out-group if 'group-specific label' tangram, 1 if 'shared label' tangram
    - so that means, for later utterances, 0.5 to 'either' group for the 'group-specific' labels but 1 to 'either' group for the 'shared' labels
- Social informativity: 0 for all earlier utterances, 1 for all later utterances
- Utterance cost: 1 for all earlier utterances, 0 for all later utterances

## Shared-unique experiment: 

- Referential informativity: 1 for all utterances
- Social informativity: 0 for 'shared' labels, 1 for 'group-specific' labels (NOTE: this does not consider the issue we discussed earlier where the group members have no way of knowing what labels the other group uses)
- Utterance cost: the same for all utterances