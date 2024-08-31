import random
import numpy as np
import torch
import matplotlib as mpl

def plot_settings():
    """
    set the default plot settings for matplotlib in order to edit the output in Illustrator.
    """
    mpl.rcParams['pdf.fonttype']=42
    mpl.rcParams['ps.fonttype']=42


def seed_everything(seed=42):
    """
    Set the seed for random number generation to ensure reproducibility.
    
    Parameters:
    seed (int): The seed value to use for random number generators. Default is 42.
    
    This function sets the seed for Python's built-in random module, NumPy, and PyTorch.
    It also sets the seed for CUDA operations in PyTorch to ensure reproducibility on GPUs.
    Additionally, it configures PyTorch to use deterministic algorithms for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    #torch.use_deterministic_algorithms_if_available()


def calculate_value_counts(df,column_names):
    """
    Calculate and print the value counts for specified columns in a DataFrame.
    
    Parameters:
    df (pandas.DataFrame): The DataFrame containing the data.
    column_names (list of str): List of column names for which to calculate value counts.
    
    This function prints the count of each unique value in the specified columns.
    It also prints the relative frequency of each unique value by dividing the counts by the total number of rows.
    """
    for column_name in column_names:
        print(df[column_name].value_counts())
        print(df[column_name].value_counts() / len(df))
    return None