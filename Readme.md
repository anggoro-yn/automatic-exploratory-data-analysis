# Automatic EDA
Write a configuration json file "config.json" and run a python script "main.py" to generate automatic EDA for forecast time series

## Observations:
- Most of the analysis can be applied for other kind of data. Only trends plots and acf/pacf are applied only to time series
- In the folder data there are a jupyter notebook to generate 3 differents datasets used in this notebook. But the pkl file with the data are 
not pushed
- There is a folder output_eda where are saved the plots output of the eda. This output are not pushed but can be obtained running the codes

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