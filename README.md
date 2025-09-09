# Signaling social identity in referential communication

Preprint: TBD

OSF project: https://osf.io/5j6uk/

## Code organization

- `analysis` contains the analysis scripts for each experiment
    - `analysis/paper_figs.Rmd` generates the figures for the paper
- `data` contains the preprocessed data
- `experiments` contains the stimuli and code for each experiment. 
    - `stim/convo_vids` contains the videos for each experiment, but they are not included in the repo to save space. They are on OSF at https://osf.io/5j6uk/files/osfstorage. There is one set of videos for Experiment 1, and another set for Experiments 2-4. 
- `figures` contains the figures for the paper. The outputs directly from the notebooks in `analysis` are in `figures/outputs`. The final figures are in `figures/PDF`.
- `model` contains the model fitting notebooks and outputs


### Experiment labels in the code

- `free-response` (Experiment 1): participants write their own descriptions for tangrams to an audience of varying composition
    - Preregistered Jul 17 2025, Collected Jul 17 and Aug 1 2025
- `shared-unique` (Experiment 2): participants choose between 'shared label' and 'group-specific label' tangrams, directed to a specified group or either group
    - Preregistered Oct 17 2024, Collected Dec 10 2024
- `earlier-later` (Experiment 3): participants choose between earlier labels and later labels, for the same tangram, directed to a specified group or either group
    - Preregistered Dec 7 2024, Collected Dec 8 2024
- `transparency`: deliver the utterances to naive observers 
    - Collected Dec 18 2024
- `varied_audience` (Experiment 4): participants choose between earlier and later labels, to an audience of varied audience composition
    - Preregistered and collected Jan 14 2025

## How to reproduce results

### Python dependencies

```{bash}
brew install py3cairo ffmpeg pango pkg-config  # (need to install this to run manim)
conda env create -f environment.yml
conda activate code-switching
```

### R dependencies

Open the project in RStudio and run the following code to install the dependencies

```{r}
# Install renv if not already installed
if (!requireNamespace("renv", quietly = TRUE)) {
  install.packages("renv")
}

# Initialize renv (this will restore packages from renv.lock)
renv::restore()
```

### Running the code

Run each of the preprocessing scripts in `analysis/{experiment}` to preprocess and anonymize the raw data (The raw data are not anonymized and thus not included in the repo). The output will be saved in `data/{experiment}`.

Then, run the analysis scripts in `analysis/{experiment}`. These notebooks will also output cleaned data for model fitting

Then, to fit the computational models, run the notebooks in `model/`.

Then, to generate the plots that directly go into the figures in the paper, run `analysis/paper_figs.Rmd`.

