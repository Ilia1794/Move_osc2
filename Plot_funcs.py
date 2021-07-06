import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np


def plot_two_lines(data_for_abscissa: np.ndarray, data_first: np.ndarray, data_second: np.ndarray,
                   name_plots: str = 'Plots', name_x: str = 'x', name_y: str = 'Force',
                   name_first: str = 'data_first', name_second: str = 'data_second') -> None:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=data_for_abscissa, y=data_first, mode='lines+markers', name=name_first))
    fig.add_trace(go.Scatter(x=data_for_abscissa, y=data_second, mode='lines', name=name_second))
    fig.write_html(f'Plots/{name_plots}.html')

    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.15, bottom=0.05, right=0.98, top=0.95)
    plt.title(name_plots)

    plt.xlabel(name_x)
    plt.ylabel(name_y)
    ax.plot(data_for_abscissa, data_first, "-.", lw=2, label=name_first)
    ax.plot(data_for_abscissa, data_second, lw=1, label=name_second)
    ax.grid(which='major',  # color = 'k',
            linewidth=0.5)
    ax.grid(which='minor',  # color = 'k',
            linestyle=':', linewidth=0.3)
    plt.legend(loc='best')
    plt.savefig(f"Plots/{name_plots}.pdf")
    plt.show(block = False)
