# Plot results

## Sources and spikes

Overlay the sources of several results, one row per unit:

```python
--8<-- "workflow.py:plot-sources"
```

## Results over many recordings

`plot_metric_heatmap` and `plot_metric_boxplot` take a long table (one row per unit, with
`config`, `condition`, `snr` and the value) and draw one panel per config; the
[FDSI benchmark results](../benchmarks/fdsi/report.ipynb) use both.

## Searches

`plot_search_landscape`, `plot_search_front` and `plot_search_parameters` take a search's
trials as a table (`study.trials_dataframe()`), continuing the
[Pareto example](tune-hyperparameters.md#two-objectives-a-pareto-front):

```python
from adapt_decomp.utils.plots import (
    plot_search_front,
    plot_search_landscape,
    plot_search_parameters,
)

chosen = int(trials["number"][on_front].iloc[0])  # e.g. the trial the search chose
plot_search_landscape(trials, chosen_number=chosen)
plot_search_front(trials, list(trials["number"][on_front]), chosen_number=chosen)
plot_search_parameters(trials)
```

See the [API reference](../reference/utils.md#plots) for every plot.
