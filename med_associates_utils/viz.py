from typing import List, Literal, Tuple, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

Palette = Union[str, List[Union[str, Tuple[float, float, float]]]]


class CumulativeEventsResult():
    def __init__(self) -> None:
        fig: Figure
        means: pd.DataFrame
        stats: pd.DataFrame


def plot_cumulative_events(event_df: pd.DataFrame, col: str = 'Day', col_order=None, event: str = 'rewarded_nosepoke', individual: str = 'Subject',
                           palette: Palette = None, hue = None, hue_order = None, indv_alpha: float = 0.1) -> CumulativeEventsResult:
    '''Plot cumulative number of events over time.

    Parameters:
        event_df: DataFrame of events
        col: column in the dataframe to form plot columns on
        col_order: order for the columns in the plot. If None, use the natural sorted order of unique items
        event: the event type to be plotted
        individual: key in the dataframe for indentifying individual subjects
        palette: palette of colors to be used.
        hue: key in the dataframe that will produce different colors
        hue_order: order for processing the hue semantic
        indv_alpha: alpha transparency for individual lines

    Returns:
        CumulativeEventsResult containing figure, summary dataframe
    '''
    result = CumulativeEventsResult()

    if col_order is None:
        plot_cols = sorted(event_df[col].unique())
    else:
        avail_cols = list(event_df[col].unique())
        plot_cols = [c for c in col_order if c in avail_cols]

    fig, axs = plt.subplots(1, len(plot_cols), figsize=(len(plot_cols) * 5, 5), sharey=True, sharex=True)
    result.fig = fig

    y_label = 'Cumulative ' + ' '.join([part.capitalize() for part in event.split('_')])

    if hue is not None and hue_order is None:
        hue_order = sorted(event_df[hue].unique())

    if hue_order is not None and palette is None:
        palette = sns.color_palette('colorblind', n_colors=len(hue_order))


    bins = np.arange(np.ceil(event_df['time'].max()))
    all_mean_df_items = []
    for ax, cur_col in zip(axs, plot_cols):
        condition = (event_df['event'] == event) & (event_df[col] == cur_col)
        for indv in event_df[individual].unique():
            sub_condition = condition & (event_df[individual] == indv)
            sub_data = event_df[sub_condition]

            # check if any events, if none, don't plot -- otherwise we generate a warning!
            if len(sub_data.index) > 0:
                sns.ecdfplot(data=event_df[sub_condition],
                            x='time',
                            hue=hue,
                            stat='count',
                            hue_order=hue_order,
                            palette=palette,
                            alpha=indv_alpha,
                            ax=ax,
                            legend=False)
        ax.set_title(cur_col)
        ax.set_xlabel('Time (minutes)')
        ax.set_ylabel(y_label)
        ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(base=600))
        formatter = mpl.ticker.FuncFormatter(lambda sec, pos: f'{sec / 60:0.0f}')
        ax.xaxis.set_major_formatter(formatter)
        sns.despine(ax=ax)


        # compute averaged cumulative events
        mean_df_items = []
        if hue is None:
            rows = event_df[condition]
            num_animals = len(rows[individual].unique())
            for bin in bins:
                counts = rows[rows['time'] < bin].value_counts(individual)
                mean_df_items.append({
                    col: cur_col,
                    'time': bin,
                    f'mean_{event}': counts.sum() / num_animals
                })
            mean_palette = None
            color = '#333333'

        else:
            for g, rows in event_df[condition].groupby(hue):
                num_animals = len(rows[individual].unique())
                for bin in bins:
                    counts = rows[rows['time'] < bin].value_counts(individual)
                    mean_df_items.append({
                        col: cur_col,
                        hue: g,
                        'time': bin,
                        f'mean_{event}': counts.sum() / num_animals
                    })
            mean_palette = palette
            color = None

        all_mean_df_items.extend(mean_df_items)

        sns.lineplot(data=pd.DataFrame(mean_df_items),
                     x='time',
                     y=f'mean_{event}',
                     palette=mean_palette,
                     color=color,
                     hue=hue,
                     hue_order=hue_order,
                     ax=ax,
                     legend='full' if cur_col == plot_cols[-1] else False)
        #print(True if cur_col == plot_cols[-1] else False)

    if hue is not None:
        sns.move_legend(ax, loc="upper left", bbox_to_anchor=(1, 1))

    result.means = pd.DataFrame(all_mean_df_items)

    #print(f'Stats for {day}')
    #print(stats.kstest(means['WT'], means['MT']))
    #print(stats.wilcoxon(np.array(means['MT']) - np.array(means['WT'])))
    #print()

    return result


class RasterPlotResult():
    def __init__(self) -> None:
        fig: Figure
        sort_order: dict[str, np.ndarray]
        max_rate: float


def plot_event_raster(event_df: pd.DataFrame, col: str = 'Day', col_order=None, row: str = 'Genotype', row_order=None, event: str = 'rewarded_nosepoke',
                      individual: str = 'Subject', palette: Palette = None, sort_col: str = 'Day4',
                      rate_max: Union[float, Literal['auto']] = 'auto') -> RasterPlotResult:
    '''Generate a raster plot of events over time

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
        rate_max: max rate for the ceiling of the colormap, if "auto" calculate max from the data

    Returns:
        RasterPlotResult containing figure, summary dataframe
    '''
    result = RasterPlotResult()

    if col_order is None:
        plot_cols = sorted(event_df[col].unique())
    else:
        avail_cols = list(event_df[col].unique())
        plot_cols = [c for c in col_order if c in avail_cols]

    if row_order is None:
        plot_rows = sorted(event_df[row].unique())
    else:
        avail_rows = list(event_df[row].unique())
        plot_rows = [r for r in row_order if r in avail_rows]

    fig, axs = plt.subplots(len(plot_rows), len(plot_cols), figsize=(len(plot_cols) * 5, len(plot_rows) * 2.5), sharey=False, sharex=True)
    result.fig = fig


    raster_events = {c: {r: [] for r in plot_rows} for c in plot_cols}
    raster_event_rates = {c: {r: [] for r in plot_rows} for c in plot_cols}
    for ci, c in enumerate(plot_cols):
        condition = (event_df['event'] == event) & (event_df[col] == c)
        for ri, r in enumerate(plot_rows):
            sub_condition = condition & (event_df[row] == r)
            for animal in sorted(event_df[sub_condition][individual].unique()):
                sub_sub_condition = sub_condition & (event_df[individual] == animal)
                events = event_df[sub_sub_condition]['time'].values
                rate = np.array([0] + list(1 / (np.diff(events) / 60)))

                raster_events[c][r].append(events)
                raster_event_rates[c][r].append(rate)

    sort_orders = {}
    for r, r_rates in raster_event_rates[sort_col].items():
        summaries = []
        for ai, animal_rates in enumerate(r_rates):
            #summaries.append(np.median(animal_rates))
            summaries.append(np.max(raster_events[sort_col][r][ai]))
        sort_orders[r] = np.argsort(summaries)
    result.sort_order = sort_orders

    if rate_max == 'auto': 
        max_rate = 0
        for c, c_rates in raster_event_rates.items():
            for r, r_rates in c_rates.items():
                for animal_rates in r_rates:
                    curr_max = np.max(animal_rates)
                    if max_rate < curr_max:
                        max_rate = curr_max
    else:
        max_rate = rate_max
    result.max_rate = max_rate


    # Define a diverging color palette using seaborn
    palette = sns.diverging_palette(250, 30, l=65, center="dark", as_cmap=True)
    norm = mpl.colors.Normalize(0, max_rate)  # Update normalization range


    # time to build the plot!
    for ci, c in enumerate(plot_cols):
        for ri, r in enumerate(plot_rows):
            ax = axs[ri, ci]

            events = raster_events[c][r]
            events = [events[s] for s in sort_orders[r] if s < len(events)]

            colors = [[palette(norm(r)) for r in animal] for animal in raster_event_rates[c][r]]
            colors = [colors[s] for s in sort_orders[r] if s < len(colors)]

            ax.eventplot(events, colors=colors, orientation="horizontal", zorder=.5)

            if ri == 0:
                ax.set_title(c)
            else:
                ax.set_xlabel('Time (minutes)')

            if ci == 0:
                ax.set_ylabel(r)

            ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(base=600))
            formatter = mpl.ticker.FuncFormatter(lambda sec, pos: f'{sec / 60:0.0f}')
            ax.xaxis.set_major_formatter(formatter)
            sns.despine(ax=ax, left=True)

    return result
