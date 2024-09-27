import dataclasses
from typing import List, Literal, Tuple, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from matplotlib.cm import ScalarMappable
from .common import Palette


@dataclasses.dataclass(init=False)
class CumulativeEventsResult:
    """Object to store results from `plot_cumulative_events()`"""

    fig: Figure
    means: pd.DataFrame


def plot_cumulative_events(
    event_df: pd.DataFrame,
    col: Union[str, None] = None,
    col_order: Union[List[str], None] = None,
    row: Union[str, None] = None,
    row_order: Union[List[str], None] = None,
    event: str = "rewarded_nosepoke",
    individual: Union[str, List[str]] = "Subject",
    palette: Palette = None,
    hue: Union[str, None] = None,
    hue_order: Union[List[str], None] = None,
    indv_alpha: float = 0.1,
) -> CumulativeEventsResult:
    """Plot cumulative number of events over time.

    Parameters:
        event_df: DataFrame of events
        col: column in the dataframe to form plot columns on
        col_order: order for the columns in the plot. If None, use the natural sorted order of unique items
        row: column in the dataframe to form plot rows on
        row_order: order for the rows in the plot. If None, use the natural sorted order of unique items
        event: the event type to be plotted
        individual: key in the dataframe for indentifying individual subjects
        palette: palette of colors to be used.
        hue: key in the dataframe that will produce different colors
        hue_order: order for processing the hue semantic
        indv_alpha: alpha transparency for individual lines

    Returns:
        CumulativeEventsResult containing figure, summary dataframe
    """
    result = CumulativeEventsResult()

    if col is not None:
        if col_order is None:
            plot_cols = sorted(event_df[col].unique())
        else:
            avail_cols = list(event_df[col].unique())
            plot_cols = [c for c in col_order if c in avail_cols]
    else:
        plot_cols = [None]

    if row is not None:
        if row_order is None:
            plot_rows = sorted(event_df[row].unique())
        else:
            avail_rows = list(event_df[row].unique())
            plot_rows = [r for r in row_order if r in avail_rows]
    else:
        plot_rows = [None]

    fig, axs = plt.subplots(
        len(plot_rows), len(plot_cols), figsize=(len(plot_cols) * 5, len(plot_rows) * 5), sharey=True, sharex=True, squeeze=False
    )
    result.fig = fig

    y_label = "Cumulative " + " ".join([part.capitalize() for part in event.split("_")])

    if len(plot_rows) > 1:
        fig.text(0, 0.5, y_label, rotation="vertical", ha="center", va="top", rotation_mode="anchor")

    if hue is not None and hue_order is None:
        hue_order = sorted(event_df[hue].unique())

    if hue_order is not None and palette is None:
        palette = sns.color_palette("colorblind", n_colors=len(hue_order))

    bins = np.arange(np.ceil(event_df["time"].max()))
    event_condition = event_df["event"] == event
    all_mean_df_items = []

    for row_i, cur_row in enumerate(plot_rows):
        if cur_row is not None:
            row_condition = event_condition & (event_df[row] == cur_row)
        else:
            row_condition = event_condition

        for col_i, cur_col in enumerate(plot_cols):
            if cur_col is not None:
                col_condition = row_condition & (event_df[col] == cur_col)
            else:
                col_condition = row_condition

            ax = axs[row_i, col_i]

            for indv in event_df[individual].drop_duplicates().values:
                if isinstance(individual, str):
                    sub_condition = col_condition & (event_df[individual] == indv)
                else:
                    sub_condition = col_condition & (event_df[individual] == indv).all(axis=1)
                sub_data = event_df[sub_condition]

                # check if any events, if none, don't plot -- otherwise we generate a warning!
                if len(sub_data.index) > 0:
                    sns.ecdfplot(
                        data=event_df[sub_condition],
                        x="time",
                        hue=hue,
                        stat="count",
                        hue_order=hue_order,
                        palette=palette,
                        alpha=indv_alpha,
                        ax=ax,
                        legend=False,
                    )
            if row_i == 0:
                ax.set_title(cur_col)
            ax.set_xlabel("Time (minutes)")
            if cur_row is None:
                ax.set_ylabel(y_label)
            else:
                ax.set_ylabel(cur_row)
            ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(base=600))
            formatter = mpl.ticker.FuncFormatter(lambda sec, pos: f"{sec / 60:0.0f}")
            ax.xaxis.set_major_formatter(formatter)
            sns.despine(ax=ax)

            # compute averaged cumulative events
            id_vars = {}
            if len(plot_cols) > 1:
                id_vars[col] = cur_col
            if len(plot_rows) > 1:
                id_vars[row] = cur_row

            mean_df_items: List[dict] = []
            if hue is None:
                rows = event_df[col_condition]
                num_animals = len(rows[individual].drop_duplicates().index)
                for bin in bins:
                    counts = rows[rows["time"] < bin].value_counts(individual)
                    mean_df_items.append({**id_vars, "time": bin, f"mean_{event}": counts.sum() / num_animals})
                mean_palette = None
                color = "#333333"

            else:
                for g, rows in event_df[col_condition].groupby(hue):
                    num_animals = len(rows[individual].drop_duplicates().index)
                    for bin in bins:
                        counts = rows[rows["time"] < bin].value_counts(individual)
                        mean_df_items.append({**id_vars, hue: g, "time": bin, f"mean_{event}": counts.sum() / num_animals})
                mean_palette = palette
                color = None

            all_mean_df_items.extend(mean_df_items)

            sns.lineplot(
                data=pd.DataFrame(mean_df_items),
                x="time",
                y=f"mean_{event}",
                palette=mean_palette,
                color=color,
                hue=hue,
                hue_order=hue_order,
                ax=ax,
                legend="full" if cur_col == plot_cols[-1] else False,
            )
            # print(True if cur_col == plot_cols[-1] else False)

        if hue is not None:
            sns.move_legend(ax, loc="upper left", bbox_to_anchor=(1, 1))

    result.means = pd.DataFrame(all_mean_df_items)

    fig.tight_layout()
    return result
