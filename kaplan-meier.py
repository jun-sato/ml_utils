from lifelines import KaplanMeierFitter
from utils import plot_settings
import matplotlib.pyplot as plt

def plot_kaplan_meier(data, time_col, event_col, group_col, group_name, ax=None):
    kmf = KaplanMeierFitter()
    data = data[data[group_col] == group_name]
    kmf.fit(data[time_col], data[event_col], label=group_name)
    
    if ax is None:
        fig, ax = plt.subplots()
    
    kmf.plot_survival_function(ax=ax)
    ax.set_title(group_name)
    return ax