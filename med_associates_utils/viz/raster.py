import dataclasses
import sys
from typing import List, Literal, Tuple, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from matplotlib.cm import ScalarMappable
from .common import Palette, get_colormap


@dataclasses.dataclass(init=False)
class RasterPlotResult:
    """Object to store results from `plot_event_raster()`"""

    fig: Figure
    sort_order: Union[dict[str, np.ndarray], None]
    max_rate: float
    events: dict[Union[str, None], dict[Union[str, None], List[np.ndarray]]]  # indexed as `rates[column][row][animal][event]`
    rates: dict[Union[str, None], dict[Union[str, None], List[np.ndarray]]]  # indexed as `rates[column][row][animal][event]`
    all_rates: list[float]  # flattened list of all rates


SORT_METRICS = Literal["max_rate", "median_rate", "ttf"]


def plot_event_raster(
    event_df: pd.DataFrame,
    col: Union[str, None] = None,
    col_order=None,
    row: Union[str, None] = None,
    row_order=None,
    event: str = "rewarded_nosepoke",
    individual: Union[str, List[str]] = "Subject",
    palette: Palette = None,
    sort_col: Union[str, None] = None,
    sort_metric: SORT_METRICS = "max_rate",
    sort_dir: Literal["asc", "dsc"] = "asc",
    rate_max: Union[float, Literal["auto"], str] = "auto",
) -> RasterPlotResult:
    """Generate a raster plot of events over time

    Parameters:
        event_df: DataFrame of events
        col: column in the dataframe to form plot columns on
        col_order: order for the columns in the plot. If None, use the natural sorted order of unique items
        row: column in the dataframe to form plot rows on
        row_order: order for the rows in the plot. If None, use the natural sorted order of unique items
        event: the event type to be plotted
        individual: key in the dataframe for indentifying individual subjects
        palette: palette of colors to be used.
        sort_col: col value to sort individuals
        rate_max: max rate for the ceiling of the colormap. If "auto" calculate max from the data; if float the value is taken literally, if str and ends with "%" the value is interpreted as a percentage.

    Returns:
        RasterPlotResult containing figure, summary dataframe
    """
    # create a result object to stash results as we go
    result = RasterPlotResult()

    # determine some parameters of the plot layout
    if col is None:
        plot_cols = [None]
    else:
        if col_order is None:
            plot_cols = sorted(event_df[col].unique())
        else:
            avail_cols = list(event_df[col].unique())
            plot_cols = [c for c in col_order if c in avail_cols]

    if row is None:
        plot_rows = [None]
    else:
        if row_order is None:
            plot_rows = sorted(event_df[row].unique())
        else:
            avail_rows = list(event_df[row].unique())
            plot_rows = [r for r in row_order if r in avail_rows]

    # construct a figure and axes, and store it in the result object
    fig, axs = plt.subplots(
        len(plot_rows), len(plot_cols), figsize=(len(plot_cols) * 5, len(plot_rows) * 2.5), sharey=False, sharex=True, squeeze=False
    )
    result.fig = fig

    max_time = event_df["time"].max()
    raster_events: dict[Union[str, None], dict[Union[str, None], List[np.ndarray]]] = {c: {r: [] for r in plot_rows} for c in plot_cols}
    raster_event_rates: dict[Union[str, None], dict[Union[str, None], List[np.ndarray]]] = {
        c: {r: [] for r in plot_rows} for c in plot_cols
    }
    for ci, c in enumerate(plot_cols):
        if c is None:
            condition = event_df["event"] == event
        else:
            condition = (event_df["event"] == event) & (event_df[col] == c)

        for ri, r in enumerate(plot_rows):
            if r is None:
                sub_condition = condition
            else:
                sub_condition = condition & (event_df[row] == r)

            for indv in event_df[sub_condition][individual].drop_duplicates().values:
                if isinstance(individual, str):
                    sub_sub_condition = sub_condition & (event_df[individual] == indv)
                else:
                    sub_sub_condition = sub_condition & (event_df[individual] == indv).all(axis=1)
                events = event_df[sub_sub_condition]["time"].values
                rate = np.array([0] + list(1 / (np.diff(events) / 60)))

                raster_events[c][r].append(events)
                raster_event_rates[c][r].append(rate)
    result.events = raster_events
    result.rates = raster_event_rates

    # Compute sorting orders
    if sort_col is None:
        sort_orders = None
    else:
        sort_orders = _compute_sort_order(raster_events, raster_event_rates, sort_col=sort_col, sort_metric=sort_metric, sort_dir=sort_dir)
    result.sort_order = sort_orders

    all_rates: List[float] = []
    for c, c_rates in raster_event_rates.items():
        for r, r_rates in c_rates.items():
            for animal_rates in r_rates:
                all_rates.extend(animal_rates)
    result.all_rates = all_rates

    max_rate: float
    if rate_max == "auto":
        max_rate = np.max(all_rates)

    elif isinstance(rate_max, str) and rate_max.endswith("%"):
        percent = float(rate_max.replace("%", ""))
        max_rate = float(np.percentile(all_rates, percent))

    elif isinstance(rate_max, (float, int)):
        max_rate = float(rate_max)

    else:
        raise ValueError(f'Did not understand argument for `rate_max` = "{rate_max}"')

    result.max_rate = max_rate

    # Define a diverging color palette using seaborn
    cmap = get_colormap(palette)
    norm = mpl.colors.Normalize(0, max_rate)  # Update normalization range

    # time to build the plot!
    for ci, c in enumerate(plot_cols):
        for ri, r in enumerate(plot_rows):
            ax = axs[ri, ci]

            events = raster_events[c][r]
            colors = [[cmap(norm(r)) for r in animal] for animal in raster_event_rates[c][r]]

            if sort_orders is not None:
                events = [events[s] for s in sort_orders[r] if s < len(events)]
                colors = [colors[s] for s in sort_orders[r] if s < len(colors)]

            ax.eventplot(events, colors=colors, orientation="horizontal", zorder=0.5)

            # first row
            if ri == 0:
                ax.set_title(c)

            # last row
            if ri == len(plot_rows) - 1:
                ax.set_xlabel("Time (minutes)")

            # first column
            if ci == 0:
                ax.set_ylabel(r)

            # set x-tick marks to be every 10 minutes, and format them into minutes
            ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(base=600))
            ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda sec, pos: f"{sec / 60:0.0f}"))

            # set y-tick marks to be integers only
            ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins='auto', integer=True))
            ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda i, pos: f"{int(i + 1)}"))

            # remove spines
            sns.despine(ax=ax, left=True)

    # finally, add a single colorbar to the plot
    cbar_label = " ".join([part.capitalize() for part in event.split("_")]) + " Rate"
    fig.colorbar(ScalarMappable(norm=norm, cmap=palette), ax=axs, location="right", label=cbar_label)

    return result


def _compute_sort_order(events, rates, sort_col: str, sort_metric: SORT_METRICS = "max_rate", sort_dir: Literal["asc", "dsc"] = "asc"):
    if sort_metric == 'ttf':
        max_time = np.max([z for v in events.values() for x in v.values() for y in x for z in y])

    sort_orders = {}
    for r, r_rates in rates[sort_col].items():
        summaries: List[float] = []
        for ai, animal_rates in enumerate(r_rates):
            if sort_metric == "max_rate":
                summaries.append(np.max(rates[sort_col][r][ai]))
            elif sort_metric == "median_rate":
                summaries.append(float(np.median(rates[sort_col][r][ai])))
            elif sort_metric == "ttf":
                # ttf: Time To Finish
                # i.e. sort on the max event time
                summaries.append(max_time - np.max(events[sort_col][r][ai]))
            else:
                raise ValueError('Did not understand value "{sort_metric}" for parameter "sort_metric"!')

        if sort_dir == "asc":
            sort_orders[r] = np.argsort(summaries)
        elif sort_dir == "dsc":
            sort_orders[r] = np.argsort(summaries)[::-1]
        else:
            raise ValueError('Did not understand value "{sort_dir}" for parameter "sort_dir"!')

    return sort_orders
