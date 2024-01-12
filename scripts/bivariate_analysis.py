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


""""""""""""""""""""""""""""""""""""""""""""""""" CORRELATIONS """""""""""""""""""""""""""""""""""""""""""""""""
# auxiliar function - filter correlation by threshold
def filter_correlations_by_threshold(df, threshold):
    """
    Given a dataframe and a threshold, transform all the values BELOW the threshold (in absolute value) into NaN
    Args:
        df (dataframe): dataframe with correlations (this dataframe can have null values)
        threshold (int): 

    Return
        df_threshold (dataframe): dataframe output correlations
    """   
    # if threshold is none, set it in 0
    if threshold == None:
        threshold = 0
    
    # transform values in absolute value below the theshold into nan
    mask = (df <= -threshold) | (df >= threshold)
    #df_threshold = df.mask(mask, np.nan)
    df_threshold = df.where(mask)
    return df_threshold


# auxiliar function - plot heatmap correlations
def plot_heatmap(df_corr):
    """
    Plot heatmap using the input dataframe
    It could be used to plot the correlations between differents variables

    Args
        df_corr (dataframe): dataframe with correlations to plot

    Return
        fig (figure plotly): fig of plotly with the plot generated
    """
    
    # heatmap
    fig = px.imshow(df_corr, text_auto=True, aspect="auto")
    
    # change title
    fig.update_layout(
      title_text = "Correlations",
        title_x = 0.5,
    title_font = dict(size = 28)
      )
    
    return fig


# calculate correlations between each features
def calculate_correlations_triu(df):
    """
    Given a dataframe, calculate the correlations (pearson) between all the variables in the dataframe
    Args
        df (dataframe)

    Return
        df_corr (dataframe): dataframe with correlations
        df_corr_upper(dataframe): dataframe with correltions - upper triangular matrix - round by 2 decimals
  """

    # calculate correlations
    df_corr = df.corr(method='pearson')
    
    # upper triangular matrix
    df_corr_upper = df_corr.where(np.triu(np.ones(df_corr.shape)).astype('bool'))
    
    # round 2 decimals
    df_corr = np.round(df_corr, 2)
    df_corr_upper = np.round(df_corr_upper, 2)
    
    return df_corr, df_corr_upper


# calculate correlations between each feature against the target
def calculate_correlations_target(df, target):
    """
    Given a dataframe and a target (that will be present in the dataframe) calculate the correlations of all features agains the target

    Args
        df (dataframe): dataframe
        target (string): feature target - that will be present in the dataframe
    
    Return
        df_corr (dataframe): dataframe with the correlations
    """

    # calculate correlations select only with the target
    df_corr_target = df.corr(method='pearson')[[target]]
    
    # roudn 3 decimals
    df_corr_target = np.round(df_corr_target, 3)
    
    # transpose to see in a better way
    df_corr_target = df_corr_target.T
    
    return df_corr_target
"""""""""""""""""""""""""""""""""""""""""""""""""  """""""""""""""""""""""""""""""""""""""""""""""""



""""""""""""""""""""""""""""""""""""""""""""""""" SCATTER PLOT """""""""""""""""""""""""""""""""""""""""""""""""
def plot_individual_scatter_plot(df, feature_x, feature_y, marginal_hist = False):
    """
    Create an individual scatter plot between two variables
    
    Args
        df (dataframe): input dataframe with the feature to plot in the scatter plot
        feature_x (string): name of the feature in x-axis
        feature_y (string): name of the feature in y-axis
        marginal_hist (bool): plot a histogram as marginal (feature_x and feature_y). By default in false
    
    Return
        fig (figure plotly): fig of plotly with the plot generated
    """

    # plot scatter plot
    if marginal_hist == True:
        fig = px.scatter(df, x = feature_x, y = feature_y, marginal_x = "histogram", marginal_y="histogram", trendline="ols")
        tittle_plot = f'scatter plot: {feature_x} vs {feature_y}. Marginal distributions'
    else:
        fig = px.scatter(df, x = feature_x, y = feature_y, trendline="ols")
        tittle_plot = f'scatter plot: {feature_x} vs {feature_y}'

    
    # update title
    fig.update_layout(
      title_text = tittle_plot,
      title_x = 0.5, # centrar titulo
      title_font = dict(size = 20)
    )

    return fig

def plot_features_to_target_scatter_plot_low(df, list_features, target, number_columns=2):
    """
    Create multiples plots (subplots) of the scatter plot between a list of features againts the target.
    -> All scatter plot with the same color. low resources, the pc is freeze to me doing scatter plot with different color

    Args
        df (dataframe): input dataframe with features and target to plot in the scatter plot
        list_feautures (list): list of features to plot against the target
        target (string): target to be ploted in each graph
        number_columns (int): number of columns in the subplot. by default 2 columns

    Return
        fig (figure plotly): fig of plotly with the plot generated 
    """

    # calculate number of rows (considering the number of colums passed as args)
    if (len(list_features) % number_columns) != 0:
        number_rows = (len(list_features) // number_columns) + 1
    else:
        number_rows = (len(list_features) // number_columns)

    # create fig to plot
    fig = make_subplots(rows=number_rows, cols=number_columns, subplot_titles=tuple(list_features))

    ########## for each feature plot:
    for index_feature in range(len(list_features)):
        feature = list_features[index_feature]

        # get indexes in the subplot (in plotly the indexes starts in 1)
        row = (index_feature // number_columns) + 1
        column = (index_feature % number_columns) + 1


        # get fig individual
        fig_aux = px.scatter(df, x = feature, y = target, trendline = "ols")
        
        # add scatter to fig global
        fig.add_trace(fig_aux.data[0],
            row = row,
            col = column
        )
        # add trendile to fig global
        trendline_ux = fig_aux.data[1]
        trendline_ux['marker']['color'] = '#d62728' # change color to brick red
        fig.add_trace(trendline_ux,
                     row = row,
                     col = column)
    
    # adjust the shape
    fig.update_layout(
        height = 350 * number_rows,  # largo
        width = 850 * number_columns,  # ancho
        title_text = "Compare scatters features againts a target",
        title_x=0.5,
        title_font = dict(size = 20)
    )

    return fig


def plot_features_to_target_scatter_plot(df, list_features, target, number_columns=2):
    """
    Create multiples plots (subplots) of the scatter plot between a list of features againts the target

    Args
        df (dataframe): input dataframe with features and target to plot in the scatter plot
        list_feautures (list): list of features to plot against the target
        target (string): target to be ploted in each graph
        number_columns (int): number of columns in the subplot. by default 2 columns

    Return
        fig (figure plotly): fig of plotly with the plot generated 
    """

    # calculate number of rows (considering the number of colums passed as args)
    if (len(list_features) % number_columns) != 0:
        number_rows = (len(list_features) // number_columns) + 1
    else:
        number_rows = (len(list_features) // number_columns)

    # create fig to plot
    fig = make_subplots(rows=number_rows, cols=number_columns, subplot_titles=tuple(list_features))

    ########## for each feature plot:
    for index_feature in range(len(list_features)):
        feature = list_features[index_feature]

        # get indexes in the subplot (in plotly the indexes starts in 1)
        row = (index_feature // number_columns) + 1
        column = (index_feature % number_columns) + 1


        # get fig individual
        trace = go.Scatter(
            x = df[feature],
            y = df[target],
            mode = 'markers',
            name = f'plot - {feature} vs {target}'
        )
        
        # add to fig global
        fig.add_trace(trace,
            row=row,
            col=column
        )

    # adjust the shape
    fig.update_layout(
        height = 350 * number_rows,  # largo
        width = 850 * number_columns,  # ancho
        title_text = "Compare scatters features againts a target",
        title_x=0.5,
        title_font = dict(size = 20)
    )

    return fig

def plot_all_features_scatter_plot(df, list_features, number_columns=2):
    """
    Create multiples plots (subplots) of the scatter plot between all features againts all features
    -> All scatter plot with the same color. low resources, the pc is freeze to me doing scatter plot with different color

    Args
        df (dataframe): input dataframe with features and target to plot in the scatter plot
        list_feautures (list): list of features to plot
        target (string): target to be ploted in each graph
        number_columns (int): number of columns in the subplot. by default 2 columns

    Return
        fig (figure plotly): fig of plotly with the plot generated 
    """

    ################# generate a list of tuples of each pair of features to generate a scatter plot  #####################
    # create dataframe with each cell is a tuple formed by a pari (row,column)
    df_tuple_features = pd.DataFrame(columns = df.columns.tolist(), index = df.columns.tolist())
    for column in df_tuple_features.columns:
        for index in df_tuple_features.index:
            df_tuple_features.at[index, column] = (index, column)
    df_tuple_features = df_tuple_features.where(np.triu(np.ones(df_tuple_features.shape), k=1).astype('bool'))
    
    # get a list of tuple of each pair of features to do a scatter plot
    stacked_series = df_tuple_features.stack().dropna()
    list_pair_features = list(stacked_series)


    ####################### plot #################################
    
    # calculate number of rows (considering the number of colums passed as args)
    if (len(list_pair_features) % number_columns) != 0:
        number_rows = (len(list_pair_features) // number_columns) + 1
    else:
        number_rows = (len(list_pair_features) // number_columns)

    # create fig to plot
    fig = make_subplots(rows=number_rows, cols=number_columns, 
                        subplot_titles = tuple([str(tupla) for tupla in list_pair_features]) ### title of each subplots
                       )

    ########## for each tuple of features to plot:
    for index_feature, (feature_x, feature_y) in enumerate(list_pair_features):

        # get indexes in the subplot (in plotly the indexes starts in 1)
        row = (index_feature // number_columns) + 1
        column = (index_feature % number_columns) + 1

        # get fig individual
        fig_aux = px.scatter(df, x = feature_x, y = feature_y, trendline = "ols")
        
        # add scatter to fig global
        fig.add_trace(fig_aux.data[0],
            row = row,
            col = column
        )
        # add trendile to fig global
        trendline_ux = fig_aux.data[1]
        trendline_ux['marker']['color'] = '#d62728' # change color to brick red
        fig.add_trace(trendline_ux,
                     row = row,
                     col = column)
    
    # adjust the shape
    fig.update_layout(
        height = 350 * number_rows,  # largo
        width = 850 * number_columns,  # ancho
        title_text = "Compare scatters for each feature againts each feature",
        title_x=0.5,
        title_font = dict(size = 20)
    )

    return fig

"""""""""""""""""""""""""""""""""""""""""""""""""  """""""""""""""""""""""""""""""""""""""""""""""""