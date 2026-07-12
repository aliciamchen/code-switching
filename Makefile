.DEFAULT_GOAL := help

help: ## Show this help message
	@grep -E '^[a-zA-Z0-9_-]+:.*?## ' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-28s\033[0m %s\n", $$1, $$2}'

# --- 1. Preprocessing (raw data -> data/{experiment}) -----------------------

preprocess-free-response: ## Preprocess raw data for free-response (Experiment 1)
	uv run python analysis/free-response/preprocess.py

preprocess-shared-unique: ## Preprocess raw data for shared-unique (Experiment 2)
	uv run python analysis/shared-unique/preprocess.py

preprocess-earlier-later: ## Preprocess raw data for earlier-later (Experiment 3)
	uv run python analysis/earlier-later/preprocess.py

preprocess-transparency: ## Preprocess raw data for transparency (naive observer task)
	uv run python analysis/transparency/preprocess.py

preprocess-varied_audience: ## Preprocess raw data for varied_audience (Experiment 4)
	uv run python analysis/varied_audience/preprocess.py

preprocess: preprocess-free-response preprocess-shared-unique preprocess-earlier-later preprocess-transparency preprocess-varied_audience ## Preprocess raw data for all experiments

# --- 2. Statistical analysis (data/{experiment} -> figures/outputs) ---------

analysis-free-response: ## Render the free-response statistical analysis
	Rscript -e "rmarkdown::render('analysis/free-response/free-response-analysis.Rmd')"

analysis-shared-unique: ## Render the shared-unique statistical analysis
	Rscript -e "rmarkdown::render('analysis/shared-unique/shared-unique_analysis.Rmd')"

analysis-earlier-later: ## Render the earlier-later statistical analysis
	Rscript -e "rmarkdown::render('analysis/earlier-later/earlier-later_analysis.Rmd')"

analysis-transparency: ## Render the transparency statistical analysis
	Rscript -e "rmarkdown::render('analysis/transparency/transparency-analysis.Rmd')"

analysis-varied_audience: ## Render the varied_audience statistical analysis
	Rscript -e "rmarkdown::render('analysis/varied_audience/varied_audience_analysis.Rmd')"

analysis: analysis-free-response analysis-shared-unique analysis-earlier-later analysis-transparency analysis-varied_audience ## Render statistical analyses for all experiments

# --- 3. Computational modeling (data/{experiment} -> model/*.ipynb outputs) -
# Notebook numbering follows paper experiment numbers, not the internal
# experiment labels above -- see CLAUDE.md's "Experiment Labels" section.

model-fit-shared-unique: ## Fit the shared-unique (Experiment 2) computational model
	uv run jupyter nbconvert --to notebook --execute --inplace model/fit-predict-exp-2.ipynb

model-fit-earlier-later: ## Fit the earlier-later (Experiment 3) computational model
	uv run jupyter nbconvert --to notebook --execute --inplace model/fit-predict-exp-3.ipynb

model-fit-varied_audience: ## Fit the varied_audience (Experiment 4) computational model
	uv run jupyter nbconvert --to notebook --execute --inplace model/fit-predict-exp-4.ipynb

model-fit: model-fit-shared-unique model-fit-earlier-later model-fit-varied_audience ## Fit all computational models

# --- 4. Paper figures --------------------------------------------------------

figures: ## Generate the final paper figures (combines outputs from all experiments)
	Rscript -e "rmarkdown::render('analysis/paper_figs.Rmd')"

# --- Full pipeline -----------------------------------------------------------

all: preprocess analysis model-fit figures ## Run the full pipeline end-to-end

.PHONY: help preprocess analysis model-fit figures all \
	preprocess-free-response preprocess-shared-unique preprocess-earlier-later preprocess-transparency preprocess-varied_audience \
	analysis-free-response analysis-shared-unique analysis-earlier-later analysis-transparency analysis-varied_audience \
	model-fit-shared-unique model-fit-earlier-later model-fit-varied_audience
