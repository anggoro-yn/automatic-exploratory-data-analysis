import numpy as np
import pandas as pd
import statsmodels
import seaborn as sns

import matplotlib.pyplot as plt
from statsmodels.graphics import tsaplots

# plotly
import plotly
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots



""""""""""""""""""""""""""""""""""""""""""""""""" PARALLEL """""""""""""""""""""""""""""""""""""""""""""""""
def plot_parallel_discrete_variables(df_percentile, list_features_target, target):
    """
    Plot a parallel with features discretes variables an target discrete
    
    Important the discrete variables can be a string categorical (ex 'low', 'medium', 'high').
    
    But in the parallel plot, it needs to be colored according the target and it needs to be a numerical category. This function transform it into 
    a integer categorical (ex. 1, 2, 3). This only works if the column categorical in pandas as internally defined the order in the string categories 
    (ex: 'low' < 'medium' < 'high') (pandas dtype category)

    Args
        df (dataframe): dataframe with the data
        list_features_target (list): list with the features to plot in the parallel plot and also it has to have the target
        target_discrete (string): in addition it is necesary define a string with the name of the target

    Return
        fig (figure plotly): fig of plotly with the plot generated
    """
    # generate df to plot in parallel plot. in this kind of plot duplicated values are not soported and return an error
    df_percentile_parallel = df_percentile[list_features_target].drop_duplicates()

    # transform target_discrete string into integer. using internally definition of the variable in pandas.
    # this is neccesary to color the parallel according the values of the target
    df_percentile_parallel[target] = df_percentile_parallel[target].cat.codes

    # plot
    fig = px.parallel_categories(df_percentile_parallel, 
                                 color = target, 
                                 color_continuous_scale=px.colors.sequential.Inferno)

    # change title
    fig.update_layout(
      title_text = "Parallel discrete variables",
        title_x = 0.5,
    title_font = dict(size = 28)
      )

    return fig
""""""""""""""""""""""""""""""""""""""""""""""""" PARALLEL """""""""""""""""""""""""""""""""""""""""""""""""