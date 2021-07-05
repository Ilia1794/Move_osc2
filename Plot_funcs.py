import plotly.graph_objects as go
import numpy as np


def plot_two_lines(data_for_abscissa: np.ndarray, data_first: np.ndarray, data_second: np.ndarray,
                   name_plots: str = 'Plots', name_x: str = 'x', name_y: str = 'Force') -> None:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=data_for_abscissa, y=data_first, mode='lines+markers', name='data_first'))
    fig.add_trace(go.Scatter(x=data_for_abscissa, y=data_second, mode='lines', name='data_second'))
    fig.write_html(f'Plots/{name_plots}.html')
