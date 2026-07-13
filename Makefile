.DEFAULT_GOAL := help

# The R analyses expect the R version pinned in renv.lock (4.4.x). If your
# default Rscript is a different version, point RSCRIPT at the right binary,
# e.g. make analysis RSCRIPT=/Library/Frameworks/R.framework/Versions/4.4-arm64/Resources/bin/Rscript
RSCRIPT ?= Rscript

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
# Each render's macros.tex is the up-to-date marker, so `make analysis` only
# re-renders experiments whose analysis script or inputs changed (touch the
# Rmd or delete the macros.tex to force a render). Set RUN_SENSITIVITY=false
# to skip the slow sensitivity refits while iterating, and run a full default
# render before committing regenerated outputs.

analysis/free-response/macros.tex: analysis/free-response/free-response-analysis.Rmd analysis/utils/macros.R data/free-response/selection_trials.csv data/free-response/exit_survey.csv analysis/free-response/red_social_similarity_results.csv
	$(RSCRIPT) -e "rmarkdown::render('analysis/free-response/free-response-analysis.Rmd')"

analysis-free-response: analysis/free-response/macros.tex ## Render the free-response statistical analysis

analysis/shared-unique/macros.tex: analysis/shared-unique/shared-unique_analysis.Rmd analysis/utils/macros.R data/shared-unique/selection_trials.csv data/shared-unique/exit_survey.csv
	$(RSCRIPT) -e "rmarkdown::render('analysis/shared-unique/shared-unique_analysis.Rmd')"

analysis-shared-unique: analysis/shared-unique/macros.tex ## Render the shared-unique statistical analysis

analysis/earlier-later/macros.tex: analysis/earlier-later/earlier-later_analysis.Rmd analysis/utils/macros.R data/earlier-later/selection_trials.csv data/earlier-later/exit_survey.csv
	$(RSCRIPT) -e "rmarkdown::render('analysis/earlier-later/earlier-later_analysis.Rmd')"

analysis-earlier-later: analysis/earlier-later/macros.tex ## Render the earlier-later statistical analysis

analysis/transparency/macros.tex: analysis/transparency/transparency-analysis.Rmd analysis/utils/macros.R data/transparency/selection_trials.csv
	$(RSCRIPT) -e "rmarkdown::render('analysis/transparency/transparency-analysis.Rmd')"

analysis-transparency: analysis/transparency/macros.tex ## Render the transparency statistical analysis

# The transparency render also refreshes data/transparency/means_by_label.csv,
# which the varied_audience analysis reads (hence the dependency below).
analysis/varied_audience/macros.tex: analysis/varied_audience/varied_audience_analysis.Rmd analysis/utils/macros.R data/varied_audience/selection_trials.csv data/varied_audience/exit_survey.csv analysis/transparency/macros.tex
	$(RSCRIPT) -e "rmarkdown::render('analysis/varied_audience/varied_audience_analysis.Rmd')"

analysis-varied_audience: analysis/varied_audience/macros.tex ## Render the varied_audience statistical analysis

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
	$(RSCRIPT) -e "rmarkdown::render('analysis/paper_figs.Rmd')"

# --- 5. Manuscript stats macros ----------------------------------------------
# The analyses and model fits export their statistics as LaTeX \newcommand
# macros (analysis/*/macros.tex and model/macros_*.tex). This target copies
# them into manuscript/stats/ so the paper can \input them instead of
# hardcoding any computed value. The manuscript directory is not part of the
# public repository, so this target is a no-op when it is absent.

manuscript-stats: ## Copy generated stats macros into manuscript/stats/ for the paper build
	@if [ ! -d manuscript ]; then echo "manuscript/ not present; skipping"; exit 0; fi; \
	mkdir -p manuscript/stats; \
	for f in analysis/*/macros.tex model/macros_*.tex analysis/sensitivity_table.tex analysis/re_structure_table.tex; do \
		[ -f "$$f" ] || continue; \
		case "$$f" in \
			analysis/sensitivity_table.tex|analysis/re_structure_table.tex) out="manuscript/stats/$$(basename $$f)" ;; \
			analysis/*) out="manuscript/stats/$$(basename $$(dirname $$f)).tex" ;; \
			*) out="manuscript/stats/$$(basename $$f)" ;; \
		esac; \
		cp "$$f" "$$out"; echo "  $$f -> $$out"; \
	done

# --- Full pipeline -----------------------------------------------------------

all: preprocess analysis model-fit figures manuscript-stats ## Run the full pipeline end-to-end

.PHONY: help preprocess analysis model-fit figures manuscript-stats all \
	preprocess-free-response preprocess-shared-unique preprocess-earlier-later preprocess-transparency preprocess-varied_audience \
	analysis-free-response analysis-shared-unique analysis-earlier-later analysis-transparency analysis-varied_audience \
	model-fit-shared-unique model-fit-earlier-later model-fit-varied_audience
