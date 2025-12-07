This markdown file contains a checklist for creating new figures for the paper. The goal is that we should explicitly visualize the results of the transparency experiment. Basically, these plots/analyses would help us argue that participants are trading off audience comprehension when they are pursuing a social signaling goal. Here is the list of figures to make and a set of subtasks for each figure. I want to add the generated code `paper_figs.Rmd` at the end of the `Experiment 4` section. 

After writing the code, check that the data joins work correctly and print a few rows to verify the transparency estimates are being matched properly. Make sure to test each data preparation step before moving to the next. Also, I want the new figures to generally fit the style of the other figures in `paper_figs.Rmd`.

First, load the transparency estimates: load means by label from `data/transparency/means_by_label.csv` and aggregated means from `data/transparency/means_agg.csv`.

Now here are the figures to make:

- [x] I want to make a figure showing the overall naive audience comprehension of earlier versus later labels. This is a bar plot of `means_agg.csv`, with error bars.
    - [x] The y-axis should be the proportion correct, and the x-axis should be the type of utterance (earlier or later). The y axis should be labeled "Proportion correct" and the x axis should be labeled "Utterance type".
    - [x] There should be a horizontal line at 1/6 to indicate the baseline. 
    - [x] Overlaid the bar plot, I also want to plot the itemwise transparency differences between earlier and later utterances, with a line connecting the two options for each item. This requires looking at the experiment 4 trials, finding the unique option sets, plotting the earlier (w error bars) mean on the left bar and the later (w error bars) on the right bar, and connecting the lines for each item. The points should be slightly jittered. 

- [x] Second, a figure showing the predicted listener referential success (y-axis) based on audience composition (x-axis) and goal (color), averaged across items. This would involve taking participants' choices on each trial and using the transparency data to estimate how likely a listener is to choose the correct tangram, based on the audience composition (by using estimated transparency for naive listeners and the assumption that in-group listeners will always get it right). This plot should show that participants' choices for the 'refer' condition should generally lead to higher listener referential success, and how this scales with the audience composition.
    - [x] Load experiment 4 trials from `data/varied_audience/selection_trials.csv`. Each row here is one trial.
        - [x] Rename `item_id` to `tangram_set`.
        - [x] Code response so that earlier is 1 and later is 0.
    - [x] For each trial, find the corresponding transparency estimates for both 2AFC options (option1 and option2) and the response, from the means by label, by finding the corresponding tangram_set, target tangram, earlier vs. later, and label in the means_by_label dataframe. Add the estimates (the empirical_stat column) to the experiment 4 selection trials dataframe. Name the transparency estimates `option1.transparency`, `option2.transparency`, and `response.transparency`.
    - [x] For each option, calculate the predicted listener referential success. The formula is (n_ingroup * 1 + n_outgroup * transparency) / (n_ingroup + n_outgroup). Add columns with these predictions called `option1.pred.success` and `option2.pred.success` and `response.pred.success`
    - [x] Add a column prop_naive, which is n_outgroup / (n_ingroup + n_outgroup)
    - [x] Group by prop_naive and goal, and tidyboot_mean `response.pred.succcess`.
    - [x] Make the plot. 

- [x] Third, a figure showing the likelihood of participants choosing earlier label (y-axis) based on itemwise transparency differences (x-axis), and goal (color), averaged across audiences. This plot should show more sensitivity in the 'refer' condition versus in the 'social' condition, to itemwise transparency differences.
    - [x] For each trial, compute the itemwise transparency difference (earlier option minus later option) and add a column for it to the dataframe
    - [x] Group by transparency difference and goal, exclude the trials where prop_naive is zero, and tidyboot_mean the response choice (where earlier is 1 and later is 0)
    - [x] Make the plot. 