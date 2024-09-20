from typing import Literal, Union
import pandas as pd

from med_associates_utils.session import FieldList, SessionCollection


def calculate_cumulative_events(
    df: pd.DataFrame, event: Union[str, list[str]], groupby: Union[str, list[str]], time_key: str = "time"
) -> pd.DataFrame:
    df = df.copy()  # make a copy
    df = df[df["event"] == event].sort_values(by=time_key)  # filter to only the requested event(s)
    df[f"cum_{event}"] = df.groupby(by=groupby).cumcount() + 1
    df = df.drop(columns="event")
    return df


def calculate_event_series_stats(
    sessions: SessionCollection,
    event: Union[str, list[str]],
    max_events: int,
    max_time: int,
    include_meta: FieldList = "all",
    time_key: str = "time",
    event_key: str = "event",
    form: Literal["long", "wide"] = "wide",
) -> pd.DataFrame:
    """Calculate statistics on event series
    The following are calculated:
        - latency: amount of time elapsed since the start of the trial until the first event is emitted
        - last: amount of time elapsed since the start of the trial unitl the last event is emitted
        - finished: amount of time elapsed since the start of the trial until all rewards are earned (equivalent to `last`) or the trial has completed (equivalent to `max_time`)
        - count: number of events emitted in the series
        - completed: boolean indicating if the number of emitted events is greater than or equal to `max_events`
    """
    if isinstance(event, str):
        event = [event]

    summaries = []
    for session in sessions:
        if include_meta == "all":
            meta = session.metadata
        else:
            meta = {k: v for k, v in session.metadata.items() if k in include_meta}

        for e in event:
            df = session.to_dataframe(include_arrays=[e], include_meta=[])
            if df.empty:
                summaries.append(
                    {
                        **meta,
                        "event": e,
                        "latency": max_time,  # no events were emitted, so latency will be the trial duration
                        "last": max_time,  # no events were emitted, so last will be the trial duration
                        "finished": max_time,
                        "count": 0,
                        "completed": False,
                    }
                )
            else:
                count = len(df.index)
                last = df.time.max()
                summaries.append(
                    {
                        **meta,
                        "event": e,
                        "latency": df.time.min(),  # time of the first event
                        "last": last,  # time of the last event
                        "finished": max_time
                        if count < max_events
                        else last,  # if the trial was completed, use last event time, otherwise use the max trial duration
                        "count": count,
                        "completed": count >= max_events,
                    }
                )
    summaries_df = pd.DataFrame(summaries)

    if form == "long":
        value_vars = ["latency", "last", "finished", "count", "completed"]
        id_vars = list(set(summaries_df.columns.to_list()) - set(value_vars))
        return summaries_df.melt(id_vars=id_vars, value_vars=value_vars, var_name="metric", value_name="value", ignore_index=True)
    elif form == "wide":
        return summaries_df
    else:
        raise ValueError(f'Did not understand value provided for `form` parameter. Got "{form}"')
