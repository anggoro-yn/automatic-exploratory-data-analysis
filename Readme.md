# Automatic EDA
Write a configuration json file "config.json" and run a python script "main.py" to generate automatic EDA for forecast time series


## Observations:
- Most of the analysis can be applied for other kind of data. Only trends plots and acf/pacf are applied only to time series
- In the folder data there are a jupyter notebook to generate 3 differents datasets used in this notebook. But the pkl file with the data are 
not pushed
- There is a folder output_eda where are saved the plots output of the eda. This output are not pushed but can be obtained running the codes
- There codes to do subplots that are invoked in the script main.py. But, in addition, there are codes to generate individual plots that there are not called in the script main.py (for example: there a function to plot subplots of histograms that are called in the script main.py but also there a function to plot the histogram of only one feature). **For ALL PLOTS there the version of individual plot and the version of subplot**


## Templates codes:
- There are a lot of functions that recibe a pandas dataframe and other parameters that each function needs and return a plotly figure
- Then with the plotly figure you can see in a jupyter notebok with the method fig.show() or you can saved it in a folder with the methods
fig.write_html to get a html interactive figure or fig.write_image to get a static image similar to other packages as matplotlib or seaborn
- In the script main.py the order is read the config file, then read the parameters since de config.json that you will use and finally call the function
to generate the plots to get a plotly figure and finally decide what to do (show, save html, save png, save pdf, etc)


## Run codes:
- It is very simple
- Open a console, for example anaconda prompt
- activate env that you are using. conda env list. conda activate -name
- navigate into folder where are located this repo. cd .. cd automatic-exploratory-data-analysis
- run script main.py.  -> python main.py


## Explications config.json
Explications of config.json to complete it. Important, this configuration is only for the plots that need it. In the codes a lot of more plots is generated, but this ones doens't need parameters

---

### Initial parametes

**global parameters**

"name_report": "indicate the name of the report"

"name_data_pkl": "indicate name of the file that have the data"

"target": "indicate name of target"

"list_features": "indicate list of features"

"number_columns": "indicate the numbers of columns to plot that accepted multiple columns"

**"reports_to_show"**
Indicate true/false which reports to do the plots and which reports skip


---
### Univariate analysis

**"ydata_profiling"**

"minimal": do a minimal report of ydata-profiling. always true when the dataset is huge


**"zoom_tendency:(start_date, end_date)"**: indicate the dates to plot trends. When the data is huge plot all the data could be too much

**"smooth_data"**: indicate the parameters of differents ways of smooth the data. Such as, moving average, weighted moving average and exponential moving average

**acf_pacf:lags** indicate the max number of lags to plot the autocorrelation function and partial autocorrelation function



---
### Bivariate analysis

**correlations:(threshold)**: indicate if add a threshold to show the correlations. for example, only show the correlations with value over 0.1 

**scatter_plot:(marginal)**: plot a scatter plot and a marginal histograms of each feature in the scatter plot

**correlations_features_lagged_target:(lags)**: indicate the number of lags in the features used to analyze the correlations of the features lagged againts the target

**parallel:(list_features)**:indicate the list of features to plot into a parallel plot vs the target as final step 


---
### Segmentation analysis

**segmentation_analysis**

"type": indicate type of segmentation. custom or by percentile

"var_segment": indicate feature or target to segment the data

"interval_segment": if custom segmentation, indicate the intervals of the values to generate the differents segments

"labels_segment": if custom segmentation, indicate the name of the differents segments



---
### Categorical analysis



  "categorical_analysis":{
    "percentile_transform":{
      "categories_features":{
        "features":["CMPC.SN", "CHILE.SN", "COPEC.SN", "ANDINA-B.SN", "MSFT", "AAPL", "GOOG", "TSLA","QQQ"],
        "percentile":["quartile", "quartile", "quartile", "quartile", "quartile", "quartile", "quartile", "quartile", "quartile"]
      },
      "categories_target":{
        "target":["VOO"],
        "percentile":["quartile"]
      }
    },

    "crosstab_freq_pair_features":{
      "freq_normalized": true
    },

    "crosstab_freq_target_feature":{
      "freq_normalized": true
    },

    "heatmap_multiple_features_vs_target_continuous":{
      "aggregation_target": ["mean", "std"]
    },

    "parallel":{
      "list_features":["CHILE.SN", "MSFT", "AAPL", "GOOG"]
    }
  }

}
