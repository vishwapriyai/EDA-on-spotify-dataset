# EDA-on-spotify-dataset



# Data Collection, Data Cleaning & Data Manipulation 
import numpy as np 
import pandas as pd 
from sklearn import datasets 





# Data Visualization
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns





# Data Transformation
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from scipy import stats





# Models Building 
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler





## Classification Problems
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn. ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score




## Regression Problems
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, LogisticRegression, SGDRegressor
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, AdaBoostRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score





# Unsupervised Learning: Clustering
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, Birch, MeanShift, SpectralClustering
from sklearn.metrics import adjusted_rand_score





v=pd.read_csv("D:/internship teachnook major project/tracks.csv")
v





#chech the shape
v.shape





#check the structure
v.info()





#check the head
v.head()




#check the summary statistics
v.describe().transpose()




#check the distribution
v.hist(bins = 20, color = 'green', figsize = (20, 14))



#TRAIN-TEST SPLIT
from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(v, test_size = 0.2, random_state = 42)




df = train_set.copy()




#DATA CLEANING
df.columns




df.isnull().sum()





df.duplicated().sum()




#DATA ANALYSIS
#Categorical Data
#Find the categorical variables
categorical_df = df.select_dtypes(include = 'object')

categorical_df.info()





#Check cardinality
for col in categorical_df.columns:
    print(f'{col}: {categorical_df[col].nunique()}')
    print('\n')



#Find the most popular artists
from wordcloud import WordCloud

plt.figure(figsize = (20, 14))

def visualize_word_counts(counts):
    wc = WordCloud(max_font_size=130, min_font_size=25, colormap='tab20', background_color='white', prefer_horizontal=.95, width=2100, height=700, random_state=0)
    cloud = wc.generate_from_frequencies(counts)
    plt.figure(figsize=(18,15))
    plt.imshow(cloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()




lead_artists = df['artists'].value_counts().head(20)

lead_artists





fig, ax = plt.subplots(figsize = (12, 10))

ax = sns.barplot(x = lead_artists.values, y = lead_artists.index, palette = 'rocket_r', orient = 'h', edgecolor = 'black', ax = ax)

ax.set_xlabel('Sum of Songs', c ='r', fontsize = 12, weight = 'bold')
ax.set_ylabel('Artists', c = 'r', fontsize = 12, weight = 'bold')
ax.set_title('20 Most Popular Artists in Dataset', c = 'r', fontsize = 14, weight = 'bold')

plt.show()





visualize_word_counts(lead_artists)





#Datetime Data
#Convert the column 'release_date' to datetime datatype in Python
import datetime





df.release_date = pd.to_datetime(df.release_date)





df.head()





#Find the first date in the dataset
df.release_date.min()



#Find the last date in the dataset
df.release_date.max()





#Find the number of songs per year
df["year"] = df["release_date"].dt.year




sns.displot(df["year"], discrete = True, aspect = 2, height = 7, kind = "hist", kde = True, color = 'green').set(title="Number of song per year")




#Find the most popular songs on Spotify
most_popularity = df.query('popularity > 90', inplace = False).sort_values('popularity', ascending = False)

most_popularity.head(10)




lead_songs = most_popularity[['name', 'popularity']].head(20)

lead_songs


# In[34]:


fig, ax = plt.subplots(figsize = (10, 10))

ax = sns.barplot(x = lead_songs.popularity, y = lead_songs.name, color = 'lightgreen', orient = 'h', edgecolor = 'black', ax = ax)

ax.set_xlabel('Popularity', c ='red', fontsize = 12, weight = 'bold')
ax.set_ylabel('Songs', c = 'red', fontsize = 12, weight = 'bold')
ax.set_title('20 Most Popular Songs in Dataset', c = 'red', fontsize = 14, weight = 'bold')

plt.show()




from sklearn import preprocessing

feat_cols = ['danceability', 'energy', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence']

mean_vals = pd.DataFrame(columns=feat_cols)
mean_vals = mean_vals.append(most_popularity[feat_cols].mean(), ignore_index=True)
mean_vals = mean_vals.append(df[feat_cols].mean(), ignore_index=True)

print(mean_vals)

import plotly.graph_objects as go
import plotly.offline as pyo
fig = go.Figure(
    data=[
        go.Scatterpolar(r=mean_vals.iloc[0], theta=feat_cols, fill='toself', name='Top 100'),
        go.Scatterpolar(r=mean_vals.iloc[1], theta=feat_cols, fill='toself', name='All'),
    ],
    layout=go.Layout(
        title=go.layout.Title(text='Feature comparison'),
        polar={'radialaxis': {'visible': True}},
        showlegend=True
    )
)

#pyo.plot(fig)
fig.show()




#Find the most danceable songs
most_danceable = df.sort_values(by='danceability',ascending=False).head(10)

most_danceable




#Find songs with the most energy
most_energy = df.sort_values(by='energy',ascending=False).head(10)

most_energy




#Find songs more likely to create positive feelings
most_valence = df.sort_values(by='valence',ascending=False).head(10)

most_valence




#Univariate Analysis
#Summary statistics
df.describe()



#Distribution
#Histograms
num_df = df.select_dtypes(include = 'number')




plt.style.use('seaborn')

names = list(num_df.columns)

plot_per_row = 2

f, axes = plt.subplots(round(len(names)/plot_per_row), plot_per_row, figsize = (15, 25))

y = 0;

for name in names:
    i, j = divmod(y, plot_per_row)
    sns.histplot(x=df[name], kde = True, ax=axes[i, j], color = 'purple')
    y = y + 1

plt.tight_layout()
plt.show()




#Boxplots
plt.style.use('seaborn')

names = list(num_df.columns)

plot_per_row = 2

f, axes = plt.subplots(round(len(names)/plot_per_row), plot_per_row, figsize = (15, 25))

y = 0;

for name in names:
    i, j = divmod(y, plot_per_row)
    sns.boxplot(x=df[name], ax=axes[i, j], palette = 'Set3')
    y = y + 1

plt.tight_layout()
plt.show()




#Bivariate Analysis
#Linear Correlation
plt.figure(figsize = (20, 14))

corr_matrix = df.corr()
cmap = sns.color_palette('magma')
sns.heatmap(corr_matrix, annot = True, cmap = cmap)
plt.title('Correlation between numerical features')
plt.show()




corr_matrix["popularity"].sort_values(ascending=False)




from pandas.plotting import scatter_matrix

attributes = ["popularity", "year", "loudness", "energy"]

scatter_matrix(df[attributes], figsize=(12, 8))

plt.show()




#Feature Selection

#New Features
#Create a new features called 'highly_popular', with threshold = 50. Songs with popularity over 50 is highly popular
df["highly_popular"] = pd.cut(df["popularity"],
                               bins=[0, 49, 100],
                               labels=[0, 1],
                            include_lowest=True)



df["highly_popular"].value_counts().sort_index().plot.bar(rot=0, grid=True)
plt.xlabel("Popularity category")
plt.ylabel("Number of songs")
plt.show()




df['highly_popular'].value_counts()





df = df.drop(labels = ['popularity'], axis = 1)




df.info()



#Data Split
#Split the target variable
np.random.seed(42)




X_train = df.copy()
y_train = df.pop("highly_popular")





from sklearn import linear_model




#Resampling
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler, NearMiss
from collections import Counter





ros = RandomOverSampler()

X_train, y_train = ros.fit_resample(X_train, y_train)

print(Counter(y_train))





#from sklearn.model_selection import train_test_split

#X_train, X_test, y_train, y_test = train_test_split(X_ros, y_ros, train_size = 0.8, test_size = 0.2, random_state = 0)



df2 = test_set.copy()





df2["highly_popular"] = pd.cut(df2["popularity"],
                               bins=[0, 49, 100],
                               labels=[0, 1],
                            include_lowest=True)





X_test = df2.copy()
y_test = df2.pop("highly_popular")





#Pipeline Building
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn import set_config

set_config(display='diagram')





def monkey_patch_get_signature_names_out():
    """Monkey patch some classes which did not handle get_feature_names_out()
       correctly in Scikit-Learn 1.0.*."""
    from inspect import Signature, signature, Parameter
    import pandas as pd
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline, Pipeline
    from sklearn.preprocessing import FunctionTransformer, StandardScaler

    default_get_feature_names_out = StandardScaler.get_feature_names_out

    if not hasattr(SimpleImputer, "get_feature_names_out"):
      print("Monkey-patching SimpleImputer.get_feature_names_out()")
      SimpleImputer.get_feature_names_out = default_get_feature_names_out

    if not hasattr(FunctionTransformer, "get_feature_names_out"):
        print("Monkey-patching FunctionTransformer.get_feature_names_out()")
        orig_init = FunctionTransformer.__init__
        orig_sig = signature(orig_init)

        def __init__(*args, feature_names_out=None, **kwargs):
            orig_sig.bind(*args, **kwargs)
            orig_init(*args, **kwargs)
            args[0].feature_names_out = feature_names_out

        __init__.__signature__ = Signature(
            list(signature(orig_init).parameters.values()) + [
                Parameter("feature_names_out", Parameter.KEYWORD_ONLY)])

        def get_feature_names_out(self, names=None):
            if callable(self.feature_names_out):
                return self.feature_names_out(self, names)
            assert self.feature_names_out == "one-to-one"
            return default_get_feature_names_out(self, names)

        FunctionTransformer.__init__ = __init__
        FunctionTransformer.get_feature_names_out = get_feature_names_out

monkey_patch_get_signature_names_out()





def column_ratio(X):
    return X[:, [0]] / (1000*60)

def ratio_name(function_transformer, feature_names_in):
    return ["ratio"]  # feature names out

def ratio_pipeline():
    return make_pipeline(
        SimpleImputer(strategy="median"),
        FunctionTransformer(column_ratio, feature_names_out=ratio_name),
        RobustScaler())




num_attribs = ['explicit', 'danceability', 'energy', 'key', 'loudness', 
               'mode', 'speechiness', 'acousticness', 'instrumentalness',
              'liveness', 'valence', 'tempo', 'time_signature']

cat_attribs = []

# log_attribs = ['duration_ms']

num_pipeline = make_pipeline(SimpleImputer(strategy="median"),
                             RobustScaler())

cat_pipeline = make_pipeline(SimpleImputer(strategy="most_frequent"),
                             OneHotEncoder(handle_unknown="ignore"))

log_pipeline = make_pipeline(SimpleImputer(strategy="median"), 
                             FunctionTransformer(np.log, feature_names_out="one-to-one"),
                             RobustScaler())

preprocessing = ColumnTransformer([
#         ("log", log_pipeline, log_attribs),
    ("duration", ratio_pipeline(), ["duration_ms"]),
        ("cat", cat_pipeline, cat_attribs),
        ('num', num_pipeline, num_attribs)],
        remainder='drop')





preprocessing




#Model Building
#Create a printing results function

def print_score(classifier, X_train, y_train, X_test, y_test):
        
    # Training set
    
    print('\n\n')

    print("TRAINING RESULTS:\n")

    # Predict
    y_train_pred = classifier.predict(X_train)

    # Evaluation
    print(f'Classification Report:\n{classification_report(y_train, y_train_pred, digits = 4)}\n')
    
    print(f'ROC AUC Score: {roc_auc_score(y_train, y_train_pred)}\n')

    print(f'Confusion Matrix:\n{confusion_matrix(y_train, y_train_pred)}\n')
    
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15,6))
    ax[0].set_title("train")
    ax[1].set_title("test")
    
    print(sns.heatmap(confusion_matrix(y_train, y_train_pred), annot=True, fmt="g", annot_kws={"size": 16}, ax=ax[0]))
    
    print('\n\n')
    
    # Test set

    print("TEST RESULTS:\n")

    # Predict
    y_test_pred = classifier.predict(X_test)

    # Evaluation
    print(f'Classification Report:\n{classification_report(y_test, y_test_pred, digits = 4)}\n')

    print(f'ROC AUC Score: {roc_auc_score(y_test, y_test_pred)}\n')

    print(f'Confusion Matrix:\n{confusion_matrix(y_test, y_test_pred)}\n')
    
    print(sns.heatmap(confusion_matrix(y_test, y_test_pred), annot=True, fmt="g", annot_kws={"size": 16}, ax=ax[1]))
    
    print('\n\n')



from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV



import warnings
warnings.filterwarnings('ignore')

from sklearn.exceptions import ConvergenceWarning




#Logistic Regression
'''Sometimes, you can see useful differences in performance or convergence with different solvers (solver).

Regularization (penalty) can sometimes be helpful.

The C parameter controls the penality strength, which can also be effective.
'''
from sklearn.linear_model import LogisticRegression




classifier = Pipeline([
    ("preprocessing", preprocessing),
    ("logistic_regression", LogisticRegression()),
])

classifier.fit(X_train,y_train)



classifier.named_steps["logistic_regression"].get_params()




print_score(classifier, X_train, y_train, X_test, y_test)




classifier = Pipeline([
    ("preprocessing", preprocessing),
    ("logistic_regression", LogisticRegression(C=100, penalty = 'l2', solver = 'lbfgs', max_iter = 1000)),
])

classifier.fit(X_train,y_train)




print_score(classifier, X_train, y_train, X_test, y_test)




feature_names = ['duration', 'explicit', 'danceability', 'energy', 'key', 'loudness', 
               'mode', 'speechiness', 'acousticness', 'instrumentalness',
              'liveness', 'valence', 'tempo', 'time_signature']





importances = pd.DataFrame(data={
    'Attribute': feature_names,
    'Importance': classifier.named_steps["logistic_regression"].coef_[0]
})

importances = importances.sort_values(by='Importance', ascending=False)

importances




plt.figure(figsize = (14, 7))

plt.bar(x=importances['Attribute'], height = importances['Importance'], color='#087E8B')
plt.title('Feature importances obtained from coefficients', size=20)
plt.xticks(rotation=30)
plt.show()




#Random Forest
from sklearn.ensemble import RandomForestClassifier




classifier = Pipeline([("preprocessing", preprocessing),("random_forest", RandomForestClassifier(random_state = 42))])

classifier.fit(X_train,y_train)




classifier.named_steps["random_forest"].get_params()




print_score(classifier, X_train, y_train, X_test, y_test)



classifier = Pipeline([("preprocessing", preprocessing),("random_forest", RandomForestClassifier(n_estimators = 300,max_features = 'sqrt',max_depth = 50,min_samples_leaf = 3,min_samples_split = 2,criterion = 'entropy',bootstrap = True,random_state = 42))])

classifier.fit(X_train,y_train)




print_score(classifier, X_train, y_train, X_test, y_test)



classifier.named_steps["random_forest"].feature_importances_




features_importance = pd.DataFrame(
    {
        'Column': feature_names,
        'Feature importance': classifier.named_steps["random_forest"].feature_importances_
    }
).sort_values('Feature importance', ascending = False)

sns.set(font_scale = 2)
fig, ax = plt.subplots(figsize = (7, 10))
ax = sns.barplot(x = "Feature importance", y = "Column", data = features_importance, palette = "Set2", orient = 'h');




#XGBOOST
from xgboost import XGBClassifier




classifier = Pipeline([
    ("preprocessing", preprocessing),
    ("xgboost", XGBClassifier()),
])

classifier.fit(X_train,y_train)




classifier.named_steps["xgboost"].get_params()





print_score(classifier, X_train, y_train, X_test, y_test)




classifier = Pipeline([
    ("preprocessing", preprocessing),
    ("xgboost", XGBClassifier(n_estimators = 100,
                            gamma = 0.5,
                            max_depth = 6,
                            learning_rate = 0.1,
                            min_child_weight = 1,
                            subsample = 1,
                            colsample_bytree = 1,
                            objective = 'binary:logistic',
                              random_state = 42)),
])

classifier.fit(X_train,y_train)




print_score(classifier, X_train, y_train, X_test, y_test)




features_importance = pd.DataFrame(
    {
        'Column': feature_names,
        'Feature importance': classifier.named_steps["xgboost"].feature_importances_
    }
).sort_values('Feature importance', ascending = False)

sns.set(font_scale = 2)
fig, ax = plt.subplots(figsize = (7, 10))
ax = sns.barplot(x = "Feature importance", y = "Column", data = features_importance, palette = "Set2", orient = 'h');






