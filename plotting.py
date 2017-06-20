"""Partly taken from https://github.com/numenta/NAB/blob/master/nab/plot.py"""

import plotly.offline as offline
from plotly.graph_objs import Data, Layout, Line, Marker, Scatter, XAxis, YAxis, Figure

offline.init_notebook_mode(connected=True)
py = offline


def createLayout(title=None, xLabel="Date", yLabel="Metric", fontSize=12, width=800, height=500):
    """Return plotly Layout object."""
    layoutArgs = {
        "title": title,
        "font": {"size": fontSize},
        "showlegend": False,
        "width": width,
        "height": height,
        "xaxis": XAxis(
            title=xLabel,
        ),
        "yaxis": YAxis(
            title=yLabel,
            domain=[0, 1],
            autorange=True,
            autotick=True,
        ),
        "barmode": "stack",
        "bargap": 0}
    margins = {"l": 70, "r": 30, "b": 50, "t": 90, "pad": 4}
    layoutArgs["margin"] = margins
    return Layout(**layoutArgs)


def get_data_scatter(df, value_key):
    return Scatter(x=df.index, y=df[value_key], name=value_key, line=Line(width=1.5), showlegend=False)


def get_anomaly_scatter(df, detections, value_key):
    return Scatter(x=detections.index,
                   y=[df.loc[index, value_key] for index in detections.index],
                   mode="markers",
                   name="Detected Anomaly",
                   text=["anomalous data"],
                   marker=Marker(
                       color="rgb(200, 20, 20)",
                       size=15.0,
                       symbol='circle',
                       line=Line(
                           color="rgb(200, 20, 20)",
                           width=2
                       )
                   ))


def get_labels_scatter(data_df, labels_df, value_key):
    anomaly_indices = labels_df[labels_df[value_key] > 0].index
    return Scatter(x=anomaly_indices,
                   y=[data_df.loc[index, value_key] if index in data_df.index else 0 for index in anomaly_indices],
                   mode="markers",
                   name="Anomaly",
                   text=["Anomalous Instance"],
                   marker=Marker(
                       color="rgb(20, 200, 20)",
                       size=15.0,
                       symbol='diamond',
                       line=Line(
                           color="rgb(20, 200, 20)",
                           width=2
                       )
                   ))


def plot(data_df, scores_df, value_key, labels_df=None, threshold=0.5, plot_notebook=True):
    """
    Plot the detected anomalies of one dimension
    :param data_df: DataFrame containing measurements
    :param scores_df: DataFrame containing anomaly scores
    :param value_key: Key of dimension to focus on
    :param labels_df: DataFrame containing the anomaly ground truth
    :param threshold: Above what value a score should be plotted as anomaly
    :param plot_notebook: Whether to plot in a Jupyter notebook
    """
    traces = [get_data_scatter(data_df, value_key)]
    if labels_df is not None:
        traces.append(get_labels_scatter(data_df, labels_df, value_key))
    detections = scores_df[scores_df[value_key] >= threshold]
    detection_scatter = get_anomaly_scatter(data_df, detections, value_key)
    traces.append(detection_scatter)
    data = Data(traces)
    layout = createLayout(title=value_key)
    fig = Figure(data=data, layout=layout)
    if plot_notebook:
        offline.iplot(fig)
    else:
        offline.plot(fig)
