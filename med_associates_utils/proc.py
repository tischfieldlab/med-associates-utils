from typing import Literal, Union
import pandas as pd



def calculate_cumulative_events(df: pd.DataFrame, event: Union[str, list[str]], groupby: Union[str, list[str]], time_key: str = 'time') -> pd.DataFrame:
    df = df.copy()  # make a copy
    df = df[df['event'] == event].sort_values(by=time_key)  # filter to only the requested event(s)
    df[f'cum_{event}'] = df.groupby(by=groupby).cumcount() + 1
    df = df.drop(columns='event')
    return df


def calculate_event_series_stats(df: pd.DataFrame, event: Union[str, list[str]], groupby: Union[str, list[str]], max_events: int, max_time: int, time_key: str = 'time', event_key: str = 'event', form: Literal['long', 'wide'] = 'wide') -> pd.DataFrame:
    ''' Calculate statistics on event series
        The following are calculated:
            - latency: amount of time elapsed since the start of the trial until the first event is emitted
            - last: amount of time elapsed since the start of the trial unitl the last event is emitted
            - finished: amount of time elapsed since the start of the trial unitl all rewards are earned (equivalent to `last`) or the trial has completed (equivalent to `max_time`)
            - count: number of events emitted in the series
            - completed: boolean indicating if the number of emitted events is greater than or equal to `max_events`
    '''
    if isinstance(event, str):
        event = [event]

    metrics = {
        'latency': (time_key, 'min'),
        'last': (time_key, 'max'),
        'finished': (time_key, 'max'),
        'count': (time_key, 'count'),
    }

    dfs = []
    for evt in event:
        subset = df[df[event_key] == evt]
        stats = subset.groupby(by=groupby).aggregate(**metrics).reset_index()
        dfs.append(stats)

    complete = pd.concat(dfs).reset_index(drop=True)

    complete['completed'] = complete['count'] >= max_events
    complete.loc[~complete['completed'], 'finished'] = max_time

    if form == 'long':
        value_vars = ['latency', 'last', 'finished', 'count', 'completed']
        id_vars = list(set(complete.columns.to_list()) - set(value_vars))
        return complete.melt(id_vars=id_vars, value_vars=value_vars, var_name='metric', value_name='value', ignore_index=True)
    elif form == 'wide':
        return complete
    else:
        raise ValueError(f'Did not understand value provided for `form` parameter. Got "{form}"')


    return complete
