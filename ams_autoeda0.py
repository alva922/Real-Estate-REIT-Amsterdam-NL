#https://medium.com/@navamisunil174/exploratory-data-analysis-of-breast-cancer-survival-prediction-dataset-c423e4137e38
#https://medium.com/illumination/how-to-perform-statistical-analysis-using-python-the-ultimate-guide-9458ae0ace1c
#https://medium.com/illumination/how-to-conduct-an-effective-exploratory-data-analysis-eda-fa4e65ab7735
#https://medium.com/gdsc-babcock-dataverse/20-of-numpy-functions-that-data-scientists-use-80-of-the-time-1634ec27d95e

import pandas as pd

listings = pd.read_csv('listings.csv')
listings.head()

filename = "listings.csv"
target_variable = "name"

#Load Autoviz
from autoviz import AutoViz_Class
%matplotlib inline

AV = AutoViz_Class()
Imported v0.1.804. After importing autoviz, you must run '%matplotlib inline' to display charts inline.
    AV = AutoViz_Class()
    dfte = AV.AutoViz(filename, sep=',', depVar='', dfte=None, header=0, verbose=1, lowess=False,
               chart_format='svg',max_rows_analyzed=150000,max_cols_analyzed=30, save_plot_dir=None)
dft = AV.AutoViz(
    filename,
    sep=",",
    depVar=target_variable,
    dfte=None,
    header=0,
    verbose=2,
    lowess=False,
    chart_format="svg",
    max_rows_analyzed=500,
    max_cols_analyzed=20,
    save_plot_dir=None
)

from autoviz import FixDQ
fixdq = FixDQ()

# importing sweetviz
import sweetviz as sv
#analyzing the dataset
advert_report = sv.analyze(listings)
#display the report
advert_report.show_html('listings_sv.html')

listings1 = pd.read_csv('listings.csv',low_memory=False)
listings2 = pd.read_csv('listings_details.csv',low_memory=False)
calendar1 = pd.read_csv('calendar.csv',low_memory=False)
neighb=pd.read_csv('neighbourhoods.csv',low_memory=False)
reviews1 = pd.read_csv('reviews.csv',low_memory=False)
reviews2 = pd.read_csv('reviews_details.csv',low_memory=False)

#https://blog.jupyter.org/make-your-pandas-or-polars-dataframes-interactive-with-itables-2-0-c64e75468fe6
from itables import show
show(listings1, buttons=["copyHtml5", "csvHtml5", "excelHtml5"])

prices1 = pd.read_csv('HousingPrices-Amsterdam-August-2021.csv',low_memory=False)
prices1.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 924 entries, 0 to 923
Data columns (total 8 columns):
 #   Column      Non-Null Count  Dtype  
---  ------      --------------  -----  
 0   Unnamed: 0  924 non-null    int64  
 1   Address     924 non-null    object 
 2   Zip         924 non-null    object 
 3   Price       920 non-null    float64
 4   Area        924 non-null    int64  
 5   Room        924 non-null    int64  
 6   Lon         924 non-null    float64
 7   Lat         924 non-null    float64
dtypes: float64(3), int64(3), object(2)
memory usage: 57.9+ KB


