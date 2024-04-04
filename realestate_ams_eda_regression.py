#https://wp.me/pdMwZd-97B
import numpy as np
import pandas as pd
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt

#Basic Python imports and installations

import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

#to make the interactive maps
import folium
from folium.plugins import FastMarkerCluster
import geopandas as gpd
from branca.colormap import LinearColormap

#to make the plotly graphs
import plotly.graph_objs as go
import chart_studio.plotly as py
from plotly.offline import iplot, init_notebook_mode
import cufflinks
cufflinks.go_offline(connected=True)
init_notebook_mode(connected=True)

#text mining
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from wordcloud import WordCloud

#Reading the datasets that belong to Project 1
listings1 = pd.read_csv('listings.csv',low_memory=False)
listings2 = pd.read_csv('listings_details.csv',low_memory=False)
calendar1 = pd.read_csv('calendar.csv',low_memory=False)
neighb=pd.read_csv('neighbourhoods.csv',low_memory=False)
reviews1 = pd.read_csv('reviews.csv',low_memory=False)
reviews2 = pd.read_csv('reviews_details.csv',low_memory=False)
#Making our Pandas DataFrames interactive with ITables 2.0
from itables import init_notebook_mode

init_notebook_mode(all_interactive=True)
from itables import show
show(listings1, buttons=["copyHtml5", "csvHtml5", "excelHtml5"])

#Basic Statistical Data Analysis
ddf = pd.read_csv("listings.csv", index_col= "id")
print(ddf.shape)
(20030, 15)
print(ddf.columns)
Index(['name', 'host_id', 'host_name', 'neighbourhood_group', 'neighbourhood',
       'latitude', 'longitude', 'room_type', 'price', 'minimum_nights',
       'number_of_reviews', 'last_review', 'reviews_per_month',
       'calculated_host_listings_count', 'availability_365'],
      dtype='object')
ddf.describe().T
ddf1 = pd.read_csv('HousingPrices-Amsterdam-August-2021.csv')
ddf1.head()
print(ddf1.shape)
(924, 8)
print(ddf1.columns)
Index(['Unnamed: 0', 'Address', 'Zip', 'Price', 'Area', 'Room', 'Lon', 'Lat'], dtype='object')
ddf1.describe().T
#Exploratory Data Analysis (EDA) & ML
listings = pd.read_csv("listings.csv", index_col= "id")
target_columns = ["property_type", "accommodates", "first_review", "review_scores_value", "review_scores_cleanliness", "review_scores_location", "review_scores_accuracy", "review_scores_communication", "review_scores_checkin", "review_scores_rating", "maximum_nights", "listing_url", "host_is_superhost", "host_about", "host_response_time", "host_response_rate", "street", "weekly_price", "monthly_price", "market"]
listings = pd.merge(listings, listings_details[target_columns], on='id', how='left')
listings = listings.drop(columns=['neighbourhood_group'])
listings['host_response_rate'] = pd.to_numeric(listings['host_response_rate'].str.strip('%'))
feq=listings['neighbourhood'].value_counts().sort_values(ascending=True)
feq.plot.barh(figsize=(10, 8), color='b', width=1)
plt.title("Number of listings by neighbourhood", fontsize=20)
plt.xlabel('Number of listings', fontsize=12)
plt.show()
lats2018 = listings['latitude'].tolist()
lons2018 = listings['longitude'].tolist()
locations = list(zip(lats2018, lons2018))

map1 = folium.Map(location=[52.3680, 4.9036], zoom_start=11.5)
FastMarkerCluster(data=locations).add_to(map1)
map1
freq = listings['room_type']. value_counts().sort_values(ascending=True)
freq.plot.barh(figsize=(15, 3), width=1, color = ["g","b","r"])
plt.show()
listings.property_type.unique()
array(['Apartment', 'Townhouse', 'Houseboat', 'Bed and breakfast', 'Boat',
       'Guest suite', 'Loft', 'Serviced apartment', 'House',
       'Boutique hotel', 'Guesthouse', 'Other', 'Condominium', 'Chalet',
       'Nature lodge', 'Tiny house', 'Hotel', 'Villa', 'Cabin',
       'Lighthouse', 'Bungalow', 'Hostel', 'Cottage', 'Tent',
       'Earth house', 'Campsite', 'Castle', 'Camper/RV', 'Barn',
       'Casa particular (Cuba)', 'Aparthotel'], dtype=object)
prop = listings.groupby(['property_type','room_type']).room_type.count()
prop = prop.unstack()
prop['total'] = prop.iloc[:,0:3].sum(axis = 1)
prop = prop.sort_values(by=['total'])
prop = prop[prop['total']>=100]
prop = prop.drop(columns=['total'])

prop.plot(kind='barh',stacked=True, color = ["r","b","g"],
              linewidth = 1, grid=True, figsize=(15,8), width=1)
plt.title('Property types in Amsterdam', fontsize=18)
plt.xlabel('Number of listings', fontsize=14)
plt.ylabel("")
plt.legend(loc = 4,prop = {"size" : 13})
plt.rc('ytick', labelsize=13)
plt.show()
feq=listings['accommodates'].value_counts().sort_index()
feq.plot.bar(figsize=(10, 8), color='b', width=1, rot=0)
plt.title("Accommodates (number of people)", fontsize=20)
plt.ylabel('Number of listings', fontsize=12)
plt.xlabel('Accommodates', fontsize=12)
plt.show()
feq = listings[listings['accommodates']==2]
feq = feq.groupby('neighbourhood')['price'].mean().sort_values(ascending=True)
feq.plot.barh(figsize=(10, 8), color='b', width=1)
plt.title("Average daily price for a 2-persons accommodation", fontsize=20)
plt.xlabel('Average daily price (Euro)', fontsize=12)
plt.ylabel("")
plt.show()
adam = gpd.read_file("neighbourhoods.geojson")
feq = pd.DataFrame([feq])
feq = feq.transpose()
adam = pd.merge(adam, feq, on='neighbourhood', how='left')
adam.rename(columns={'price': 'average_price'}, inplace=True)
adam.average_price = adam.average_price.round(decimals=0)

map_dict = adam.set_index('neighbourhood')['average_price'].to_dict()
color_scale = LinearColormap(['yellow','red'], vmin = min(map_dict.values()), vmax = max(map_dict.values()))

def get_color(feature):
    value = map_dict.get(feature['properties']['neighbourhood'])
    return color_scale(value)

map3 = folium.Map(location=[52.3680, 4.9036], zoom_start=11)
folium.GeoJson(data=adam,
               name='Amsterdam',
               tooltip=folium.features.GeoJsonTooltip(fields=['neighbourhood', 'average_price'],
                                                      labels=True,
                                                      sticky=False),
               style_function= lambda feature: {
                   'fillColor': get_color(feature),
                   'color': 'black',
                   'weight': 1,
                   'dashArray': '5, 5',
                   'fillOpacity':0.5
                   },
               highlight_function=lambda feature: {'weight':3, 'fillColor': get_color(feature), 'fillOpacity': 0.8}).add_to(map3)
map3
fig = plt.figure(figsize=(20,10))
plt.rc('xtick', labelsize=16)
plt.rc('ytick', labelsize=20)

ax1 = fig.add_subplot(121)
feq = listings[listings['number_of_reviews']>=10]
feq1 = feq.groupby('neighbourhood')['review_scores_location'].mean().sort_values(ascending=True)
ax1=feq1.plot.barh(color='b', width=1)
plt.title("Average review score location (at least 10 reviews)", fontsize=20)
plt.xlabel('Score (scale 1-10)', fontsize=20)
plt.ylabel("")

ax2 = fig.add_subplot(122)
feq = listings[listings['accommodates']==2]
feq2 = feq.groupby('neighbourhood')['price'].mean().sort_values(ascending=True)
ax2=feq2.plot.barh(color='b', width=1)
plt.title("Average daily price for a 2-persons accommodation", fontsize=20)
plt.xlabel('Average daily price (Euro)', fontsize=20)
plt.ylabel("")

plt.tight_layout()
plt.show()
sum_available = calendar[calendar.available == "t"].groupby(['date']).size().to_frame(name= 'available').reset_index()
sum_available['weekday'] = sum_available['date'].dt.day_name()
sum_available = sum_available.set_index('date')

sum_available.iplot(y='available', mode = 'lines', xTitle = 'Date', yTitle = 'number of listings available',\
                   text='weekday', title = 'Number of listings available by date')
#Let's consider the Amsterdam House Price Prediction project, including data processing, EDA, feature engineering, and ML regression.
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
df = pd.read_csv('HousingPrices-Amsterdam-August-2021.csv')
df.head()
sns.heatmap(df.corr())
df = df.dropna(axis = 0, inplace = False)
sns.boxplot(x='Price', data = df)
q1 = df.describe()['Price']['25%']
q3 = df.describe()['Price']['75%']
iqr = q3 - q1
max_price = q3 + 1.5 * iqr 
outliers = df[df['Price'] >= max_price]
outliers_count = outliers['Price'].count()
df_count = df['Price'].count()
print('Percentage removed: ' + str(round(outliers_count/df_count * 100, 2)) + '%')
Percentage removed: 7.72%
df= df[df['Price'] <= max_price]
sns.boxplot(x='Price', data = df)
df['Zip No'] = df['Zip'].apply(lambda x:x.split()[0])
df['Letters'] = df['Zip'].apply(lambda x:x.split()[-1])
def word_separator(string):
    list = string.split()
    word = []
    number = [] 
    for element in list:
        if element.isalpha() == True: 
            word.append(element)
        else:
            break
    word = ' '.join(word)
    return word
df['Street'] = df['Address'].apply(lambda x:word_separator(x))
numerical = ['Price', 'Area', 'Room', 'Lon', 'Lat']
categorical = ['Address', 'Zip No', 'Letters', 'Street']
from sklearn.preprocessing import LabelEncoder
for c in categorical:
    lbl = LabelEncoder() 
    lbl.fit(list(df[c].values)) 
    df[c] = lbl.transform(list(df[c].values))
df.drop(['Zip', 'Unnamed: 0', '
Address'], axis =1, inplace = True)
sns.heatmap(df.corr())
#Preparing data for training and testing supervised ML models
from sklearn.model_selection import train_test_split
X = df.drop('Price', axis =1)
y = df['Price']
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.4)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)   
from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
linreg.fit(X_train, y_train)
predictions = linreg.predict(X_test)
plt.scatter(y_test,predictions)
plt.title('Linear Regression',fontsize=18)
plt.xlabel('Test Data',fontsize=18)
plt.ylabel('Prediction',fontsize=18)
from sklearn.linear_model import Lasso
lasso = Lasso()
lasso.fit(X_train, y_train)
predictions = lasso.predict(X_test)
plt.scatter(y_test,predictions)
plt.title('Lasso',fontsize=18)
plt.xlabel('Test Data',fontsize=18)
plt.ylabel('Prediction',fontsize=18)
from sklearn.linear_model import ElasticNet
elasticnet = ElasticNet()
elasticnet.fit(X_train, y_train)
predictions = elasticnet.predict(X_test)
plt.scatter(y_test,predictions)
plt.title('ElasticNet',fontsize=18)
plt.xlabel('Test Data',fontsize=18)
plt.ylabel('Prediction',fontsize=18)
from sklearn.linear_model import Ridge
ridge = Ridge()
ridge.fit(X_train, y_train)
predictions = ridge.predict(X_test)
plt.scatter(y_test,predictions)
plt.title('Ridge',fontsize=18)
plt.xlabel('Test Data',fontsize=18)
plt.ylabel('Prediction',fontsize=18)
from sklearn.ensemble import RandomForestRegressor
random_forest = RandomForestRegressor()
random_forest.fit(X_train, y_train)
predictions = random_forest.predict(X_test)
plt.scatter(y_test,predictions)
plt.title('Random Forest',fontsize=18)
plt.xlabel('Test Data',fontsize=18)
plt.ylabel('Prediction',fontsize=18)
from xgboost import XGBRegressor
xgb = XGBRegressor()
xgb.fit(X_train, y_train)
predictions = xgb.predict(X_test)
plt.scatter(y_test,predictions)
plt.title('XGBoost',fontsize=18)
plt.xlabel('Test Data',fontsize=18)
plt.ylabel('Prediction',fontsize=18)
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

random_grid = {'bootstrap': [True, False],
 'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
 'max_features': ['auto', 'sqrt'],
 'min_samples_leaf': [1, 2, 4],
 'min_samples_split': [2, 5, 10],
 'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]}

random_cv = RandomizedSearchCV(estimator = random_forest, param_distributions = random_grid, n_iter = 100, cv = 10, verbose = 2, n_jobs = -1)
random_cv.fit(X_train, y_train)
Fitting 10 folds for each of 100 candidates, totalling 1000 fits
RandomizedSearchCV(cv=10, estimator=RandomForestRegressor(), n_iter=100,
                   n_jobs=-1,
                   param_distributions={'bootstrap': [True, False],
                                        'max_depth': [10, 20, 30, 40, 50, 60,
                                                      70, 80, 90, 100, None],
                                        'max_features': ['auto', 'sqrt'],
                                        'min_samples_leaf': [1, 2, 4],
                                        'min_samples_split': [2, 5, 10],
                                        'n_estimators': [200, 400, 600, 800,
                                                         1000, 1200, 1400, 1600,
                                                         1800, 2000]},
                   verbose=2)
random_cv.best_params_ 
{'n_estimators': 1400,
 'min_samples_split': 2,
 'min_samples_leaf': 2,
 'max_features': 'sqrt',
 'max_depth': 100,
 'bootstrap': False}

param_grid = {'bootstrap': [True, False],
'max_depth': [60,65,70,75,80],
'min_samples_leaf':[1,2,3],
'min_samples_split': [1,2,3],
'n_estimators': [1750,1760,1770,1780,1790,1800,1810,1820,1830,1840,1850]}
grid_search = GridSearchCV(estimator = random_forest, param_grid = param_grid, cv = 3, n_jobs = -1, verbose = 2)

grid_search.fit(X_train,y_train)

Fitting 3 folds for each of 990 candidates, totalling 2970 fits
GridSearchCV(cv=3, estimator=RandomForestRegressor(), n_jobs=-1,
             param_grid={'bootstrap': [True, False],
                         'max_depth': [60, 65, 70, 75, 80],
                         'min_samples_leaf': [1, 2, 3],
                         'min_samples_split': [1, 2, 3],
                         'n_estimators': [1750, 1760, 1770, 1780, 1790, 1800,
                                          1810, 1820, 1830, 1840, 1850]},
             verbose=2)

grid_search.best_params_
{'bootstrap': True,
 'max_depth': 65,
 'min_samples_leaf': 1,
 'min_samples_split': 2,
 'n_estimators': 1840}
tuned_random_forest = RandomForestRegressor(n_estimators = 1750, max_depth = 80, min_samples_leaf = 1, min_samples_split = 2)
random_forest.fit(X_train, y_train)
predictions = random_forest.predict(X_test)

cv = cross_val_score(tuned_random_forest, X_train, y_train, cv=20, scoring = 'neg_mean_squared_error')
print("The Random Forest Regressor with tuned parameters has a RMSE of: " + str(abs(cv.mean())**0.5))
The Random Forest Regressor with tuned parameters has a RMSE of: 95830.77429927936
# Calculate testing scores
from sklearn.metrics import mean_absolute_error, r2_score
test_mae = mean_absolute_error(y_test, predictions)
test_mse = mean_squared_error(y_test, predictions)
test_rmse = mean_squared_error(y_test, predictions, squared=False)
test_r2 = r2_score(y_test, predictions)
print(test_mae)
65426.81317647059
print(test_mse)
9152415095.08142
print(test_rmse)
95668.25541986966
print(test_r2)
0.8319334007165564

#Project 1: SweetViz AutoEDA
listings = pd.read_csv('listings.csv')
# importing sweetviz
import sweetviz as sv
#analyzing the dataset
advert_report = sv.analyze(listings)
#display the report
advert_report.show_html('listings_sv.html')
#Project 1: AutoViz AutoEDA
filename = "listings.csv"
target_variable = "name"
#Load Autoviz
from autoviz import AutoViz_Class
%matplotlib inline

AV = AutoViz_Class()
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
#Project 1: Geospatial EDA
import numpy as np
import pandas as pd
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt

import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

#to make the interactive maps
import folium
from folium.plugins import FastMarkerCluster
import geopandas as gpd
from branca.colormap import LinearColormap

#to make the plotly graphs
import plotly.graph_objs as go
import chart_studio.plotly as py
from plotly.offline import iplot, init_notebook_mode
import cufflinks
cufflinks.go_offline(connected=True)
init_notebook_mode(connected=True)

#text mining
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from wordcloud import WordCloud
from json.decoder import JSONDecoder
import warnings


warnings.simplefilter(action='ignore', category=FutureWarning)


import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud


nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('omw-1.4', quiet=True)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english') + ['ha', 'wa', 'br', 'b'])
def preprocess(text):
    text = list(filter(str.isalpha, word_tokenize(text)))
    text = list(lemmatizer.lemmatize(word) for word in text)
    text = list(word for word in text if word not in stop_words)
    return ' '.join(text)


def draw_wordcloud(texts, max_words=1000, width=1000, height=500):
    wordcloud = WordCloud(background_color='white', max_words=max_words,
                          width=width, height=height)
    joint_texts = ' '.join(list(texts))
    wordcloud.generate(joint_texts)
    return wordcloud.to_image()


def draw_choropleth(neighbourhoods_geojson,feature, stat='mean', only_reviewed=False, title=None):
    stats = {
        'mean': pd.api.typing.DataFrameGroupBy.mean,
        'median': pd.api.typing.DataFrameGroupBy.median,
        'sum': pd.api.typing.DataFrameGroupBy.sum,
    }
    gb = stats[stat]((listings_reviewed if only_reviewed else listings).groupby(by='neighbourhood', as_index=False), feature)
    fig = px.choropleth_mapbox(gb, geojson=neighbourhoods_geojson, color=feature,
                           locations="neighbourhood", featureidkey="properties.neighbourhood",
                           center={"lat": 52.3676, "lon": 4.9041}, title=title or f'{feature} by neighbourhood',
                           mapbox_style="carto-positron", zoom=10, opacity=0.5)
    return fig.show()
  calendar = pd.read_csv('calendar.csv')
listings = pd.read_csv('listings.csv')
listings_detailed = pd.read_csv('listings_details.csv')
reviews = pd.read_csv('reviews.csv')
reviews_detailed = pd.read_csv('reviews_details.csv')
neighbourhoods = pd.read_csv('neighbourhoods.csv')
listings = listings_detailed.drop(columns=['neighbourhood']).rename(columns={'neighbourhood_cleansed': 'neighbourhood'})  # we will only need neighbourhood_cleansed
reviews = reviews_detailed
for column in ['host_since', 'first_review', 'last_review']:
    listings[column] = pd.to_datetime(listings[column], format='%Y-%m-%d')
    listings[column].dt.day.describe()

calendar.date = pd.to_datetime(calendar.date, format='%Y-%m-%d')

listings.price = listings.price.replace('[\$,]', '', regex=True).astype(float)
calendar.price = calendar.price.replace('[\$,]', '', regex=True).astype(float)
fig = px.histogram(listings, x="neighbourhood", category_orders={'neighbourhood': list(listings.neighbourhood.value_counts().index)}, title='Number of Listings by Neighbourhood')
fig.show()
adam = gpd.read_file("neighbourhoods.geojson")
adam.head()
gb = listings.neighbourhood.value_counts().reset_index()
gb.head()
gb = listings.neighbourhood.value_counts().reset_index()
fig = px.choropleth_mapbox(gb, geojson=adam, color='neighbourhood',
                       locations="index", featureidkey="properties.neighbourhood",
                       center={"lat": 52.3676, "lon": 4.9041}, title=f'Size by Neighbourhood',
                       mapbox_style="carto-positron", zoom=10, opacity=0.5)
fig.show()
top10_neighbourhoods = list(listings.neighbourhood.value_counts()[:10].index)
top10_neighbourhoods
['De Baarsjes - Oud-West',
 'De Pijp - Rivierenbuurt',
 'Centrum-West',
 'Centrum-Oost',
 'Westerpark',
 'Zuid',
 'Oud-Oost',
 'Bos en Lommer',
 'Oostelijk Havengebied - Indische Buurt',
 'Oud-Noord']
listings_reviewed = listings[listings.number_of_reviews > 0]
listings_reviewed.loc[:, 'first_review_year'] = listings_reviewed['first_review'].dt.year
gb = listings_reviewed.groupby(by=['first_review_year', 'neighbourhood'], as_index=False).size()
gb.first_review_year = gb.first_review_year.astype(int)
fig = px.choropleth_mapbox(gb, geojson=adam, color='size',
                       locations="neighbourhood", featureidkey="properties.neighbourhood",
                       center={"lat": 52.3676, "lon": 4.9041}, title='Neighbourhoods Yearly Growth',
                       mapbox_style="carto-positron", zoom=10, opacity=0.5, animation_frame='first_review_year')
fig.show()
res = listings_reviewed.copy()
for year in range(2009, 2023):
    listings_at_year = listings_reviewed[listings_reviewed.first_review_year == year]
    ls = [res]
    for future_year in range(year + 1, 2024):
        l = listings_at_year.copy()
        l.first_review_year = future_year
        ls.append(l)
    res = pd.concat(ls)
fig = px.scatter_mapbox(res.sort_values('first_review_year', ascending=True), lat='latitude', lon='longitude', center={"lat": 52.3676, "lon": 4.9041}, #color="peak_hour", size="car_hours",
                        zoom=10, mapbox_style="carto-positron", animation_frame='first_review_year', opacity=0.25, title='Neighbourhoods Cumulative Yearly Growth')
fig.show()
draw_choropleth(adam,'number_of_reviews', 'sum', title='Total Number of Reviews by Neighbourhood (~Total Tourists Volume)')
listings_reviewed['lifetime_in_months'] = ((listings_reviewed.last_review - listings_reviewed.first_review)/np.timedelta64(1, 'D'))/30 + 1/30
listings_reviewed['load'] = listings_reviewed['number_of_reviews'] / np.ceil(listings_reviewed['lifetime_in_months']
draw_choropleth(adam,'load', 'sum', only_reviewed=True, title='Total Number of Reviews Per Month by Neighbourhood (~Monthly Airbnb Guests Volume)')
draw_choropleth(adam,'load', 'median', only_reviewed=True, title='Median Number of Reviews Per Month by Neighbourhood (~Listing Busyness)')
draw_choropleth(adam,'price', 'median', title='Median Price by Neighbourhood')  
listings_in_top10_neighbourhoods = listings[listings.neighbourhood.isin(top10_neighbourhoods)]
len(listings_in_top10_neighbourhoods)
16952
fig = px.violin(listings_in_top10_neighbourhoods, y="price", x="neighbourhood", log_y=False, range_y=[-10, 2000], points="all", box=True, title='Price Distribution by Neighbourhood',
               category_orders={'neighbourhood': list(listings_in_top10_neighbourhoods.groupby('neighbourhood')['price'].aggregate('median').reset_index().sort_values(by='price')['neighbourhood'])}
)
fig.show()     
draw_choropleth(adam,'review_scores_location', 'median', title='Median Location Scores by Neighbourhood')
fig = px.violin(listings, y="review_scores_location", x="neighbourhood", box=True, points="all", range_y=[1.8, 10.5], title='Location Score Distribution by Neighbourhood',
               category_orders={'neighbourhood': list(listings.groupby('neighbourhood')['review_scores_location'].aggregate('mean').reset_index().sort_values(by='review_scores_location')['neighbourhood'])}
)
fig.show()
fig = px.histogram(listings, x='price', nbins=1000, barmode='group', range_x=[0, 500], histnorm='probability', title='Price Distribution')
fig.show()
fig = px.histogram(listings, x='price', nbins=100, barmode='group', range_x=[0, 500], histnorm='probability', title='Price Distribution')
fig.show()
listings_reviewed.loc[:, 'number_of_reviews_jittered'] = listings_reviewed.number_of_reviews + np.exp(np.random.randn(len(listings_reviewed)) / 10)
listings_reviewed.loc[:, 'review_scores_rating_jittered'] = listings_reviewed.review_scores_rating + np.random.randn(len(listings_reviewed)) / 10
fig = px.scatter(listings_reviewed[listings_reviewed.price < 1000], x='number_of_reviews_jittered', y='review_scores_rating_jittered', color='lifetime_in_months', log_x=True, opacity=0.25, marginal_x='box', marginal_y='histogram', title='Rating by Number of Reviews')
fig.show()
aspect_scores_feats = ['review_scores_accuracy', 'review_scores_cleanliness', 'review_scores_checkin', 'review_scores_communication', 'review_scores_location', 'review_scores_value']
scores_feats = ['review_scores_rating'] + aspect_scores_feats
px.imshow(listings[scores_feats].corr(), title='Review Scores Correlations')
listings_reviewed.loc[:, 'load_jittered'] = listings_reviewed.load + np.random.randn(len(listings_reviewed)) / 5
listings_reviewed.loc[:, 'price_jittered'] = listings_reviewed.price + np.random.randn(len(listings_reviewed)) * 2

px.histogram(listings_reviewed, x='price', y='load', histfunc='avg', range_x=[0, 300], range_y=[0, 4], nbins=2500, title='Number of Reviews per Month by Price').show()
px.scatter(listings_reviewed, x='lifetime_in_months', y='number_of_reviews', color='price', range_color=[0, 300], range_y=[0, 600], opacity=0.5, title='Number of Reviews by Active Lifetime').show()
fig = px.histogram(listings_reviewed, x='lifetime_in_months', nbins=int(listings_reviewed.lifetime_in_months.max()), barmode='group', marginal='box', title='Active Lifetime In Months')
fig.show()
fig = px.scatter(listings, x='first_review', y='last_review', marginal_x='histogram', marginal_y='histogram', title='Listings Lifetimes Scatter')
fig.show()
fig = px.histogram(listings, x='property_type', color='room_type', barmode='group', title='Listing Types')
fig.show()
fig = px.box(listings, x='accommodates', y='price', range_y=[0, 1000], range_x=[0, 9], title='Price by Capacity')
fig.show()
#Project 1: NLP Wordcloud Images
listings['listing_name'] = listings.name.astype('string')
print(listings['listing_name'])
0                 Quiet Garden View Room & Super Fast WiFi
1                        Quiet apt near center, great view
2               100%Centre-Studio 1 Private Floor/Bathroom
3                      Lovely apt in City Centre (Jordaan)
4        Romantic, stylish B&B houseboat in canal district
                               ...                        
20025     Family House City + free Parking+garden (160 m2)
20026                    Home Sweet Home in Indische Buurt
20027               Amsterdam Cozy apartment nearby center
20028              Home Sweet Home for a Guest or a Couple
20029          Cosy two bedroom appartment near 'de Pijp'!
Name: listing_name, Length: 20000, dtype: string
txt=listings['listing_name'].str.cat(sep=' ')  
# Create and generate a word cloud image:

wordcloud = WordCloud().generate(txt)
type(txt)
str
# Display the generated image:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
STOPWORDS = nltk.corpus.stopwords.words('english')
# Create stopword list:
stopwords = set(STOPWORDS)
stopwords.update(["Amsterdam", "city", "Beautiful","centre",'center'])

# Generate a word cloud image
wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(txt)

# Display the generated image:
# the matplotlib way:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
reviews = reviews.dropna()
reviews.loc[:, 'length'] = reviews.comments.str.len()
fig = px.histogram(reviews[reviews.length != 0], x='length', nbins=1000, barmode='group', title='Length of Reviews')
fig.show()
reviews = reviews[reviews.length < 750]
reviews_with_rating = reviews.join(listings[['id', 'review_scores_rating']].set_index('id'), on='listing_id', validate='m:1')
txt1=reviews_with_rating['comments'].str.cat(sep=' ')
#wordcloud = WordCloud().generate(txt1)
wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(txt1)
# Display the generated image:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
#Project 1: ML Regression of Review Scores
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from pprint import pprint

def calc_prediction_quality(X_feats, y_feat):
    df = listings[X_feats + [y_feat]].dropna(how='any')
    X_train, X_test, y_train, y_test = train_test_split(df[X_feats], df[y_feat], test_size=0.33, random_state=42)
    lr = LinearRegression().fit(X_train, y_train)
    mse = mean_squared_error(y_test, lr.predict(X_test))
    from sklearn.metrics import r2_score
    r2score=r2_score(y_test, lr.predict(X_test))
    print(f'LR R2: {r2score}')
    print(f'constant model MSE: {mean_squared_error(y_test, [y_train.mean()] * len(y_test))}')
    print(f'LR MSE: {mse}')
    print(f'LR intercept: {lr.intercept_}')
    print(f'LR weights:')
    pprint(dict(zip(aspect_scores_feats, lr.coef_)))

calc_prediction_quality(aspect_scores_feats, 'review_scores_rating')

LR R2: 0.6843609885371096
constant model MSE: 42.109935110705806
LR MSE: 13.289090701988293
LR intercept: 0.1273328699656986
LR weights:
{'review_scores_accuracy': 2.6342633776024216,
 'review_scores_checkin': 0.8067556404716549,
 'review_scores_cleanliness': 2.159631791937603,
 'review_scores_communication': 1.8040139905123578,
 'review_scores_location': 0.31715042347693845,
 'review_scores_value': 2.2077211691735097}
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from pprint import pprint

def calc_prediction_quality(X_feats, y_feat):
    df = listings[X_feats + [y_feat]].dropna(how='any')
    X_train, X_test, y_train, y_test = train_test_split(df[X_feats], df[y_feat], test_size=0.33, random_state=42)
    lr = RandomForestRegressor(n_estimators=1000,max_depth=22).fit(X_train, y_train)
    mse = mean_squared_error(y_test, lr.predict(X_test))
    from sklearn.metrics import r2_score
    r2score=r2_score(y_test, lr.predict(X_test))
    print(f'RF R2: {r2score}')
    print(f'RF constant model MSE: {mean_squared_error(y_test, [y_train.mean()] * len(y_test))}')
    print(f'RF MSE: {mse}')

calc_prediction_quality(aspect_scores_feats, 'review_scores_rating')
RF R2: 0.6564884792280963
RF constant model MSE: 42.109935110705806
RF MSE: 14.462584125956381
from xgboost import XGBRegressor
print(xgboost.__version__)
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def calc_prediction_quality(X_feats, y_feat):
    df = listings[X_feats + [y_feat]].dropna(how='any')
    X_train, X_test, y_train, y_test = train_test_split(df[X_feats], df[y_feat], test_size=0.33, random_state=42)
    model = XGBRegressor(n_estimators=1000, max_depth=17, eta=0.1, subsample=0.7, colsample_bytree=0.8)
    lr = model.fit(X_train, y_train)
    mse = mean_squared_error(y_test, lr.predict(X_test))
    from sklearn.metrics import r2_score
    r2score=r2_score(y_test, lr.predict(X_test))
    print(f'XGB R2: {r2score}')
    print(f'XGB constant model MSE: {mean_squared_error(y_test, [y_train.mean()] * len(y_test))}')
    print(f'XGB MSE: {mse}')

calc_prediction_quality(aspect_scores_feats, 'review_scores_rating')
2.0.3
XGB R2: 0.6217210860475758
XGB constant model MSE: 42.109935110705806
XGB MSE: 15.926367196706325
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def calc_prediction_quality(X_feats, y_feat):
    df = listings[X_feats + [y_feat]].dropna(how='any')
    X_train, X_test, y_train, y_test = train_test_split(df[X_feats], df[y_feat], test_size=0.4, random_state=42)
    model = SVR(C=28.0, epsilon=0.2)
    lr = model.fit(X_train, y_train)
    mse = mean_squared_error(y_test, lr.predict(X_test))
    from sklearn.metrics import r2_score
    r2score=r2_score(y_test, lr.predict(X_test))
    print(f'SVR R2: {r2score}')
    print(f'SVR constant model MSE: {mean_squared_error(y_test, [y_train.mean()] * len(y_test))}')
    print(f'SVR MSE: {mse}')

calc_prediction_quality(aspect_scores_feats, 'review_scores_rating')
SVR R2: 0.6371308133892497
SVR constant model MSE: 42.77398907443572
SVR MSE: 15.518963670618406
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def calc_prediction_quality(X_feats, y_feat):
    df = listings[X_feats + [y_feat]].dropna(how='any')
    X_train, X_test, y_train, y_test = train_test_split(df[X_feats], df[y_feat], test_size=0.4, random_state=42)
    model = DecisionTreeRegressor(max_depth=28)
    lr = model.fit(X_train, y_train)
    mse = mean_squared_error(y_test, lr.predict(X_test))
    from sklearn.metrics import r2_score
    r2score=r2_score(y_test, lr.predict(X_test))
    print(f'DT R2: {r2score}')
    print(f'DT constant model MSE: {mean_squared_error(y_test, [y_train.mean()] * len(y_test))}')
    print(f'DT MSE: {mse}')

calc_prediction_quality(aspect_scores_feats, 'review_scores_rating')
DT R2: 0.6202272475226304
DT constant model MSE: 42.77398907443572
DT MSE: 16.2418848616904
import numpy as np
import matplotlib.pyplot as plt 
 
  
# creating the dataset
data = {'LR':0.684, 'RF':0.656, 'XGB':0.621, 
        'SVR':0.637,'DT':0.620}
courses = list(data.keys())
values = list(data.values())
  
fig = plt.figure(figsize = (10, 5))
 
# creating the bar plot
plt.bar(courses, values, color ='maroon', 
        width = 0.4)
 
plt.xlabel("Regressor")
plt.ylabel("R2-Score")
plt.title("R2-Score of 5 Regression Models")
plt.show()
#Project 2: Tuned RF Regression of Prices
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
df = pd.read_csv('HousingPrices-Amsterdam-August-2021.csv')
df = df.dropna(axis = 0, inplace = False)
q1 = df.describe()['Price']['25%']
q3 = df.describe()['Price']['75%']
iqr = q3 - q1
max_price = q3 + 1.5 * iqr 
outliers = df[df['Price'] >= max_price]
outliers_count = outliers['Price'].count()
df_count = df['Price'].count()
print('Percentage removed: ' + str(round(outliers_count/df_count * 100, 2)) + '%')
Percentage removed: 7.72%
df= df[df['Price'] <= max_price]
df['Zip No'] = df['Zip'].apply(lambda x:x.split()[0])
df['Letters'] = df['Zip'].apply(lambda x:x.split()[-1])
def word_separator(string):
    list = string.split()
    word = []
    number = [] 
    for element in list:
        if element.isalpha() == True: 
            word.append(element)
        else:
            break
    word = ' '.join(word)
    return word
df['Street'] = df['Address'].apply(lambda x:word_separator(x))
numerical = ['Price', 'Area', 'Room', 'Lon', 'Lat']
categorical = ['Address', 'Zip No', 'Letters', 'Street']
from sklearn.preprocessing import LabelEncoder
for c in categorical:
    lbl = LabelEncoder() 
    lbl.fit(list(df[c].values)) 
    df[c] = lbl.transform(list(df[c].values))

df.drop(['Zip', 'Unnamed: 0', 'Address'], axis =1, inplace = True)
from sklearn.model_selection import train_test_split
X = df.drop('Price', axis =1)
y = df['Price']
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

random_grid = {'bootstrap': [True, False],
 'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
 'max_features': ['auto', 'sqrt'],
 'min_samples_leaf': [1, 2, 4],
 'min_samples_split': [2, 5, 10],
 'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]}

random_cv = RandomizedSearchCV(estimator = random_forest, param_distributions = random_grid, n_iter = 100, cv = 10, verbose = 2, n_jobs = -1)
random_cv.fit(X_train, y_train)
Fitting 10 folds for each of 100 candidates, totalling 1000 fits
random_cv.best_params_ 

{'n_estimators': 1600,
 'min_samples_split': 2,
 'min_samples_leaf': 1,
 'max_features': 'sqrt',
 'max_depth': 70,
 'bootstrap': False}
tuned_random_forest = RandomForestRegressor(n_estimators = 1600, max_depth = 70, min_samples_leaf = 1, min_samples_split = 2)
random_forest.fit(X_train, y_train)
predictions = random_forest.predict(X_test)
cv = cross_val_score(tuned_random_forest, X_train, y_train, cv=20, scoring = 'neg_mean_squared_error')
print("The Random Forest Regressor with tuned parameters has a RMSE of: " + str(abs(cv.mean())**0.5))
The Random Forest Regressor with tuned parameters has a RMSE of: 92683.47375648536

# Calculate testing scores
from sklearn.metrics import mean_absolute_error, r2_score
test_mae = mean_absolute_error(y_test, predictions)
test_mse = mean_squared_error(y_test, predictions)
test_rmse = mean_squared_error(y_test, predictions, squared=False)
test_r2 = r2_score(y_test, predictions)
print(test_mae)
60217.95247058822
print(test_mse)
7459469611.399326
print(test_rmse)
86368.22107349048
print(test_r2)
0.8544722257795427
plt.scatter(y_test,predictions)
plt.title('Tuned Random Forest',fontsize=18)
plt.xlabel('Test Data',fontsize=18)
plt.ylabel('Prediction',fontsize=18)
# Generate data
x = y_test
y = predictions

# Initialize layout
fig, ax = plt.subplots(figsize = (9, 9))

# Add scatterplot
ax.scatter(x, y, s=60, alpha=0.7, edgecolors="k")

# Fit linear regression via least squares with numpy.polyfit
# It returns an slope (b) and intercept (a)
# deg=1 means linear fit (i.e. polynomial of degree 1)
b, a = np.polyfit(x, y, deg=1)

# Create sequence 
xseq = x

# Plot regression line
ax.plot(xseq, a + b * xseq, color="r", lw=4);
plt.xlabel('Test Data',fontsize=18)
plt.ylabel('Prediction',fontsize=18)
plt.title('Tuned Random Forest with Regression Line',fontsize=22)
                
