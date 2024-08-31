import numpy as np
import matplotlib.pyplot as plt
from utils import plot_settings
from scipy import stats

def bland_altman_plot(y_true, y_pred, column_name1='var1', column_name2='var2', title=None, ax=None, output_path='./output.pdf'):
    """
    Plot a Bland-Altman plot to compare two sets of measurements.
    
    Parameters:
    y_true (array-like): The true measurements.
    y_pred (array-like): The predicted measurements.
    column_name1 (str): The name of the true measurements column.
    column_name2 (str): The name of the predicted measurements column.
    title (str): The title of the plot. Default is 'Bland-Altman plot'.
    ax (matplotlib.axes.Axes): The axes on which to plot the data. If None, a new figure and axes will be created.
    
    Returns:
    matplotlib.axes.Axes: The axes on which the data was plotted.
    
    This function creates a Bland-Altman plot to compare two sets of measurements.
    The plot displays the mean of the true and predicted measurements on the x-axis
    and the difference between the true and predicted measurements on the y-axis.
    The mean difference and 95% limits of agreement are also displayed on the plot.
    """
    plot_settings()
    if ax is None:
        fig, ax = plt.subplots()
    
    # Calculate the mean of the true and predicted measurements
    mean = np.mean([y_true, y_pred], axis=0)
    
    # Calculate the difference between the true and predicted measurements
    diff = y_true - y_pred
    
    # Calculate the mean difference and 95% limits of agreement
    mean_diff = np.mean(diff)
    LoA = 1.96 * np.std(diff)
    
    # Perform a t-test to calculate the p-value for the mean difference
    t_stat, p_value = stats.ttest_rel(y_true, y_pred)
    
    # Plot the Bland-Altman plot
    ax.scatter(mean, diff, edgecolors='orange', facecolors='none', label='data point')
    ax.axhline(mean_diff, color='blue', linestyle='-', label=f'Mean Difference: {mean_diff:.2f}')
    ax.axhline(mean_diff + LoA, color='orange', linestyle='-', alpha=0.5, label=f'Upper LoA: {mean_diff + LoA:.2f}')
    ax.axhline(mean_diff - LoA, color='orange', linestyle='-', alpha=0.5, label=f'Lower LoA: {mean_diff - LoA:.2f}')
    ax.axhline(mean_diff + LoA, color='black', linestyle='--')
    ax.axhline(mean_diff - LoA, color='black', linestyle='--')
    
    # Set the title and labels
    if title is not None:
        ax.set_title(title)
    ax.set_xlabel(f'Mean of {column_name1} and {column_name2}')
    ax.set_ylabel(f'{column_name1} - {column_name2}')
    ax.legend()
    print('Upper LoA: ',mean_diff+LoA,'\n',
          'Lower LoA: ',mean_diff-LoA,'\n',
          'p-value: ',p_value)
    
    # Add the p-value as text to the plot
    ax.text(0.05, 0.95, f'p-value: {p_value:.6f}', transform=ax.transAxes, verticalalignment='top')
    ax.figure.savefig(output_path)
    return ax