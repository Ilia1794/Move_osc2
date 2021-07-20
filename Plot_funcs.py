import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np


def plot_two_lines(data_for_abscissa: np.ndarray, data_first: np.ndarray, data_second: np.ndarray,
                   name_plots: str = 'Plots', name_x: str = 'x', name_y: str = 'Force',
                   name_first: str = 'data_first', name_second: str = 'data_second', mod_iter:
                   int = 10) -> None:
    absc = np.zeros((data_for_abscissa.shape[0]//10), dtype='float64')
    data_1 = np.zeros((data_for_abscissa.shape[0]//10), dtype='float64')
    data_2 = np.zeros((data_for_abscissa.shape[0]//10), dtype='float64')
    for i in range(data_for_abscissa.shape[0]//10):
        absc[i] = data_for_abscissa[i*10]
        data_1[i] = data_first[i*10]
        data_2[i] = data_second[i*10]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=absc, y=data_1, mode='lines+markers', name=name_first))
    #    go.Scatter(x=data_for_abscissa, y=data_first, mode='lines+markers', name=name_first))
    fig.add_trace(go.Scatter(x=absc, y=data_2, mode='lines', name=name_second))
    #fig.add_trace(go.Scatter(x=data_for_abscissa, y=data_second, mode='lines', name=name_second))
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
