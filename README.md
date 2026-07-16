# Signaling social identity in referential communication

Preprint: https://doi.org/10.31234/osf.io/cykfe_v3

OSF project (preregistrations and stimulus videos): https://osf.io/5j6uk/

Archived code and data: https://doi.org/10.5281/zenodo.17905933

## Code organization

- `analysis` contains the analysis scripts for each experiment
  - `analysis/paper_figs.Rmd` generates the figures for the paper
  - `analysis/{experiment}/{experiment}_analysis.Rmd` is each experiment's analysis script. The outputs of the statistical models are saved in `analysis/{experiment}/model_outputs.txt`. Each experiment's key test is accompanied by a sensitivity analysis over alternative random-effects specifications (`analysis/{experiment}/sensitivity.csv`).
  - `analysis/utils/macros.R` contains helpers that export the reported statistics as LaTeX macros (see "Where the reported statistics come from" below).
- `data` contains the preprocessed, de-identified data; `data/README.md` is a data dictionary describing every file and column
- `experiments` contains the stimuli and code for each experiment.
  - `stim/convo_vids` contains the videos for each experiment, but they are not included in the repo to save space. They are on OSF at https://osf.io/5j6uk/files/osfstorage. There is one set of videos for Experiment 1, and another set for Experiments 2-4.
- `figures` contains the figures for the paper. The outputs directly from the notebooks in `analysis` are in `figures/outputs`. The final figures in `figures/PDF` are assembled from those outputs in Adobe Illustrator (`figures/figures.ai`), which adds the schematic panels and layout. 
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

The Python packages are managed by [uv](https://docs.astral.sh/uv/).

```{bash}
brew install py3cairo ffmpeg pango pkg-config  # (need to install this to run manim for generating the videos -- but they are also in OSF)
```

```{bash}
uv sync
```

### R dependencies

The analyses are conducted with R version `4.4.2` (the version pinned in `renv.lock`). One way to install it is with [rig](https://github.com/r-lib/rig), which can also launch RStudio under that version (the `4.4-arm64` name below is for Apple Silicon; run `rig list` to see the installed name on other platforms):

```bash
rig add 4.4.2
rig rstudio 4.4-arm64
```

Then open the project in RStudio and run the following code to restore the R packages from `renv.lock`:

```{r}
# Install renv if not already installed
if (!requireNamespace("renv", quietly = TRUE)) {
  install.packages("renv")
}

# Restore packages from renv.lock
renv::restore()
```

### Running the code

Everything downstream of preprocessing runs from the de-identified data committed in `data/`. Preprocessing is the one exception: it reads the raw data, which are not anonymized and therefore not included in the repo. The committed files in `data/{experiment}` are the outputs of that stage.

The `Makefile` wraps all of the steps; run `make help` to see the available targets.

```bash
make reproduce    # reproduce all downstream results from the committed de-identified data
make analysis     # render all statistical analyses (runs from the committed data)
make model-fit    # fit the computational models
make figures      # generate the paper figures
make all          # full pipeline end-to-end (requires the raw data)
```

To reproduce results, run `make reproduce`. This runs the statistical analyses, runs the computational models, generates the paper figures, and refreshes the manuscript statistics. The `make all` target additionally runs preprocessing and therefore requires the private, non-anonymized raw data.

The analyses expect the R version pinned in `renv.lock` (4.4.x). If your default `Rscript` is a different version, point the `RSCRIPT` variable at the right binary, for example:

```bash
make reproduce RSCRIPT=/Library/Frameworks/R.framework/Versions/4.4-arm64/Resources/bin/Rscript
```

The analysis targets track their input files (each render's `macros.tex` serves as the up-to-date marker), so `make analysis` only re-renders experiments whose analysis script or data changed. 
You can set `RUN_SENSITIVITY=false` to skip the slow sensitivity refits. 

### Where the reported statistics come from

The analyses export every statistic reported in the manuscript -- demographics, regression estimates and contrasts, and model-fit parameters -- as automatically generated LaTeX macros (`analysis/{experiment}/macros.tex` for the statistical analyses and `model/macros_*.tex` for the computational models, produced by the helpers in `analysis/utils/macros.R` and `model/macros.py`). The manuscript inputs these files directly, so no computed value is hardcoded in the paper, and every reported number can be traced to the script that generated it. `make manuscript-stats` copies the macro files into the manuscript directory (which is not part of this repository).
