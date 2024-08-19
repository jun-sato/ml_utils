import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from utils import plot_settings
import matplotlib.pyplot as plt

def kaplan_meier_plot(data, time_col, event_col, group_col, threshold=None, ax=None, output_path='./output.pdf'):
    """
    Plot Kaplan-Meier curves for two groups divided by a threshold or already divided into two groups and perform log-rank test.
    
    Parameters:
    data (DataFrame): The data containing the columns.
    time_col (str): The name of the column representing time.
    event_col (str): The name of the column representing the event occurrence (1 if event occurred, 0 otherwise).
    group_col (str): The name of the column to be used for dividing into two groups or already divided into two groups.
    threshold (float, optional): The threshold value to divide the group_col into two groups. If None, assume group_col is already binary.
    ax (matplotlib.axes.Axes, optional): The axes on which to plot the data. If None, a new figure and axes will be created.
    output_path (str): The path where the plot will be saved. Default is './output.pdf'.
    
    Returns:
    matplotlib.axes.Axes: The axes on which the data was plotted.
    """
    plot_settings()
    kmf_low = KaplanMeierFitter()
    kmf_high = KaplanMeierFitter()
    data = data[[time_col, event_col, group_col]].dropna()
    
    if threshold is not None:
        # Divide the data into two groups based on the threshold
        data_low = data[data[group_col] <= threshold]
        data_high = data[data[group_col] > threshold]
    else:
        # Assume the data is already divided into two groups
        unique_values = data[group_col].unique()
        if len(unique_values) != 2:
            raise ValueError("group_col must have exactly two unique values if threshold is not provided.")
        data_low = data[data[group_col] == unique_values[0]]
        data_high = data[data[group_col] == unique_values[1]]
    
    # Fit the Kaplan-Meier estimator for each group
    kmf_low.fit(data_low[time_col], data_low[event_col], label=f'{group_col} == {unique_values[0]}' if threshold is None else f'{group_col} <= {threshold}')
    kmf_high.fit(data_high[time_col], data_high[event_col], label=f'{group_col} == {unique_values[1]}' if threshold is None else f'{group_col} > {threshold}')
    
    if ax is None:
        fig, ax = plt.subplots()
    
    # Plot the survival functions
    kmf_low.plot_survival_function(ax=ax)
    kmf_high.plot_survival_function(ax=ax)
    
    # Perform log-rank test
    results = logrank_test(data_low[time_col], data_high[time_col], event_observed_A=data_low[event_col], event_observed_B=data_high[event_col])
    p_value = results.p_value
    
    # Set the title and labels
    ax.set_title(f'Kaplan-Meier Plot by {group_col}')
    ax.set_xlabel('Time')
    ax.set_ylabel('Survival Probability')
    
    # Add the p-value as text to the plot in the lower left corner, slightly above the legend
    ax.text(0.025, 0.2, f'p-value: {p_value:.5f}', transform=ax.transAxes, verticalalignment='bottom')
    
    # Place the legend in the lower left corner
    ax.legend(loc='lower left')
    
    # Save the plot
    ax.figure.savefig(output_path)
    
    return ax