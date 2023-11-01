from typing import List, Literal, Tuple, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from matplotlib.cm import ScalarMappable

Palette = Union[str, List[Union[str, Tuple[float, float, float]]]]


class CumulativeEventsResult():
    '''Object to store results from `plot_cumulative_events()`'''

    def __init__(self) -> None:
        fig: Figure
        means: pd.DataFrame
        stats: pd.DataFrame


def plot_cumulative_events(event_df: pd.DataFrame, col: str = 'Day', col_order=None, row: str = None, row_order=None, event: str = 'rewarded_nosepoke', individual: str = 'Subject',
                           palette: Palette = None, hue = None, hue_order = None, indv_alpha: float = 0.1) -> CumulativeEventsResult:
    '''Plot cumulative number of events over time.

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
    '''
    result = CumulativeEventsResult()

    if col_order is None:
        plot_cols = sorted(event_df[col].unique())
    else:
        avail_cols = list(event_df[col].unique())
        plot_cols = [c for c in col_order if c in avail_cols]

    if row is not None:
        if row_order is None:
            plot_rows = sorted(event_df[row].unique())
        else:
            avail_rows = list(event_df[row].unique())
            plot_rows = [r for r in row_order if r in avail_rows]
    else:
        plot_rows = [None]

    fig, axs = plt.subplots(len(plot_rows), len(plot_cols), figsize=(len(plot_cols) * 5, len(plot_rows) * 5), sharey=True, sharex=True, squeeze=False)
    result.fig = fig

    y_label = 'Cumulative ' + ' '.join([part.capitalize() for part in event.split('_')])

    if len(plot_rows) > 1:
        fig.text(0, 0.5, y_label, rotation='vertical', ha='center', va='top', rotation_mode='anchor')

    if hue is not None and hue_order is None:
        hue_order = sorted(event_df[hue].unique())

    if hue_order is not None and palette is None:
        palette = sns.color_palette('colorblind', n_colors=len(hue_order))


    bins = np.arange(np.ceil(event_df['time'].max()))
    event_condition = (event_df['event'] == event)
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

            for indv in event_df[individual].unique():
                sub_condition = col_condition & (event_df[individual] == indv)
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
            if row_i == 0:
                ax.set_title(cur_col)
            ax.set_xlabel('Time (minutes)')
            if cur_row is None:
                ax.set_ylabel(y_label)
            else:
                ax.set_ylabel(cur_row)
            ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(base=600))
            formatter = mpl.ticker.FuncFormatter(lambda sec, pos: f'{sec / 60:0.0f}')
            ax.xaxis.set_major_formatter(formatter)
            sns.despine(ax=ax)


            # compute averaged cumulative events
            id_vars = {}
            if len(plot_cols) > 1:
                id_vars[col] = cur_col
            if len(plot_rows) > 1:
                id_vars[row] = cur_row

            mean_df_items = []
            if hue is None:
                rows = event_df[col_condition]
                num_animals = len(rows[individual].unique())
                for bin in bins:
                    counts = rows[rows['time'] < bin].value_counts(individual)
                    mean_df_items.append({
                        **id_vars,
                        'time': bin,
                        f'mean_{event}': counts.sum() / num_animals
                    })
                mean_palette = None
                color = '#333333'

            else:
                for g, rows in event_df[col_condition].groupby(hue):
                    num_animals = len(rows[individual].unique())
                    for bin in bins:
                        counts = rows[rows['time'] < bin].value_counts(individual)
                        mean_df_items.append({
                            **id_vars,
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

    fig.tight_layout()
    return result


class RasterPlotResult():
    '''Object to store results from `plot_event_raster()`'''

    def __init__(self) -> None:
        fig: Figure
        sort_order: dict[str, np.ndarray]
        max_rate: float
        rates: dict[str, dict[str, List[np.ndarray]]] # indexed as `rates[column][row][animal][event]`
        all_rates: list[float] # flattened list of all rates

SORT_METRICS = Literal['max_rate', 'median_rate', 'ttf']

def plot_event_raster(event_df: pd.DataFrame, col: str = 'Day', col_order=None, row: str = 'Genotype', row_order=None, event: str = 'rewarded_nosepoke',
                      individual: str = 'Subject', palette: Palette = None, sort_col: str = 'Day4', sort_metric: SORT_METRICS = 'max_rate', sort_dir: Literal['asc', 'dsc'] = 'asc',
                      rate_max: Union[float, Literal['auto'], str] = 'auto', cmap = None) -> RasterPlotResult:
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
        rate_max: max rate for the ceiling of the colormap. If "auto" calculate max from the data; if float the value is taken literally, if str and ends with "%" the value is interpreted as a percentage.

    Returns:
        RasterPlotResult containing figure, summary dataframe
    '''
    # create a result object to stash results as we go
    result = RasterPlotResult()

    # determine some parameters of the plot layout
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

    # construct a figure and axes, and store it in the result object
    fig, axs = plt.subplots(len(plot_rows), len(plot_cols), figsize=(len(plot_cols) * 5, len(plot_rows) * 2.5), sharey=False, sharex=True)
    result.fig = fig

    max_time = event_df['time'].max()
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
    result.rates = raster_event_rates

    sort_orders = {}
    for r, r_rates in raster_event_rates[sort_col].items():
        summaries = []
        for ai, animal_rates in enumerate(r_rates):
            if sort_metric == 'max_rate':
                summaries.append(np.max(raster_event_rates[sort_col][r][ai]))
            elif sort_metric == 'median_rate':
                summaries.append(np.median(raster_event_rates[sort_col][r][ai]))
            elif sort_metric == 'ttf':
                # ttf: Time To Finish
                # i.e. sort on the max event time
                summaries.append(max_time - np.max(raster_events[sort_col][r][ai]))
            else:
                raise ValueError('Did not understand value "{sort_metric}" for parameter "sort_metric"!')

        if sort_dir == 'asc':
            sort_orders[r] = np.argsort(summaries)
        elif sort_dir == 'dsc':
            sort_orders[r] = np.argsort(summaries)[::-1]
        else:
            raise ValueError('Did not understand value "{sort_dir}" for parameter "sort_dir"!')
    result.sort_order = sort_orders

    all_rates = []
    for c, c_rates in raster_event_rates.items():
        for r, r_rates in c_rates.items():
            for animal_rates in r_rates:
                all_rates.extend(animal_rates)
    result.all_rates = all_rates

    if rate_max == 'auto':
        max_rate = np.max(all_rates)

    elif isinstance(rate_max, str) and rate_max.endswith("%"):
        percent = float(rate_max.replace('%', ''))
        max_rate = np.percentile(all_rates, percent)

    elif isinstance(rate_max, (float, int)):
        max_rate = rate_max

    else:
        raise ValueError(f'Did not understand argument for `rate_max` = "{rate_max}"')

    result.max_rate = max_rate


    # Define a diverging color palette using seaborn
    if cmap is None:
        palette = sns.diverging_palette(250, 30, l=65, center="dark", as_cmap=True)
    elif isinstance(cmap, str):
        palette = mpl.colormaps[cmap]
    else:
        palette = cmap
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

            # first row
            if ri == 0:
                ax.set_title(c)

            # last row
            if ri == len(plot_rows) - 1:
                ax.set_xlabel('Time (minutes)')

            # first column
            if ci == 0:
                ax.set_ylabel(r)

            # set tick marks to be every 10 minutes
            ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(base=600))
            formatter = mpl.ticker.FuncFormatter(lambda sec, pos: f'{sec / 60:0.0f}')
            ax.xaxis.set_major_formatter(formatter)

            # remove spines
            sns.despine(ax=ax, left=True)

    # finally, add a single colorbar to the plot
    cbar_label = ' '.join([part.capitalize() for part in event.split('_')]) + ' Rate'
    fig.colorbar(ScalarMappable(norm=norm, cmap=palette), ax=axs, location='right', label=cbar_label)

    return result
