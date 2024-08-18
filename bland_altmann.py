import numpy as np
import matplotlib.pyplot as plt
from utils import plot_settings


def plot_bland_altmann(y_true, y_pred, title='Bland-Altman plot', ax=None):
    """
    Plot a Bland-Altman plot to compare two sets of measurements.
    
    Parameters:
    y_true (array-like): The true measurements.
    y_pred (array-like): The predicted measurements.
    title (str): The title of the plot. Default is 'Bland-Altman plot'.
    ax (matplotlib.axes.Axes): The axes on which to plot the data. If None, a new figure and axes will be created.
    
    Returns:
    matplotlib.axes.Axes: The axes on which the data was plotted.
    
    This function creates a Bland-Altman plot to compare two sets of measurements.
    The plot displays the mean of the true and predicted measurements on the x-axis
    and the difference between the true and predicted measurements on the y-axis.
    The mean difference and 95% limits of agreement are also displayed on the plot.
    """
    if ax is None:
        fig, ax = plt.subplots()
    
    # Calculate the mean of the true and predicted measurements
    mean = np.mean([y_true, y_pred], axis=0)
    
    # Calculate the difference between the true and predicted measurements
    diff = y_true - y_pred
    
    # Calculate the mean difference and 95% limits of agreement
    mean_diff = np.mean(diff)
    LoA = 1.96 * np.std(diff)
    
    # Plot the Bland-Altman plot
    ax.scatter(mean, diff, color='black', s=10)
    ax.axhline(mean_diff, color='red', linestyle='--', label=f'Mean Difference: {mean_diff:.2f}')
    ax.axhline(mean_diff + LoA, color='blue', linestyle='--', label=f'Upper LoA: {mean_diff + LoA:.2f}')
    ax.axhline(mean_diff - LoA, color='blue', linestyle='--', label=f'Lower LoA: {mean_diff - LoA:.2f}')
    
    # Set the title and labels
    ax.set_title(title)
    ax.set_xlabel('Mean of True and Predicted Measurements')
    ax.set_ylabel('Difference between True and Predicted Measurements')
    ax.legend()
    
    return ax