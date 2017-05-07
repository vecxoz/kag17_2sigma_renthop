
# coding: utf-8

# # Kaggle 2017: Two Sigma Connect: Rental Listing Inquiries [Competition](https://www.kaggle.com/c/two-sigma-connect-rental-listing-inquiries)
# 
# ** * Silver solution (118-th place) * **  
# ** * Original submission is [here](https://github.com/vecxoz/kag17_2sigma_renthop/tree/master/original_best_submission) * **  
# ** * Public LB: 0.50762 * **  
# ** * Private LB: 0.50665 * **  
# ** * Author: Igor Ivanov ([vecxoz](https://www.kaggle.com/vecxoz)) * **  
# ** * Email: vecxoz@gmail.com * **  
# ** * MIT License * **

# # How to Reproduce
# 
# * Submission score may slightly vary depending on versions of packages, but should be around 120-th place
# * You need 8 GB RAM
# * On machine with 4 cores solution runs about 2 hours
# * To reproduce solution basically you need Ubuntu, Java and Python 3 with  
#   NumPy, Pandas, SciPy, Scikit-learn and XGBoost (maybe I missed something)
# * You can deploy ML environment on Ubuntu for Python 3 using this [script](https://github.com/vecxoz/vecsnip/blob/master/deploy_cloud_ml_ubuntu_python_no_gpu.sh)
# * Clone (or download) repository [https://github.com/vecxoz/kag17_2sigma_renthop](https://github.com/vecxoz/kag17_2sigma_renthop). You will have dir `kag17_2sigma_renthop`
# * Put `train.json` and `test.json` files into `kag17_2sigma_renthop/data`. You can download this files from competition [data page](https://www.kaggle.com/c/two-sigma-connect-rental-listing-inquiries/data)
# * Run `$ python3 solution.py` or just run all cells of this notebook `solution.ipynb`
# * Files `solution.py` and  `solution.ipynb` contain completely identical Python code
# * After script is complete you will have submission file: `kag17_2sigma_renthop/reproduced_submission/reproduced_submission.csv`

# # Overview
# The dataset for this competition is just amazing.  
# We have all kinds of features: numerical, categorical, geospatial (lat/lon), text, pictures...  
# Just endless possibilities for feature engineering.  
# Let's look at some training example:  
# ```
# >>> train_df.iloc[12]
# bathrooms                                                   1.000000
# bedrooms                                                           2
# building_id                         67c9b420da4a365bc26a6cd0ef4a5320
# created                                          2016-04-19 05:37:25
# description        ***LOW FEE. Beautiful CHERRY OAK WOODEN FLOORS...
# display_address                                            E 38th St
# features            [Doorman, Elevator, Laundry in Building, No Fee]
# interest_level                                                  high
# latitude                                                   40.748800
# listing_id                                                   6895442
# longitude                                                 -73.977000
# manager_id                          537e06890f6a86dbb70c187db5be4d55
# photos             [https://photos.renthop.com/2/6895442_34d617a5...
# price                                                           3000
# street_address                                         137 E 38th St
# ```
# The final submission is an ensemble (weighted average) of 3 first-level models.  
# Each first-level model is meta-model by nature itself.  
# First-level models are built based on the concept of 'mixed stacking':
# * fit some model on dataset
# * predict dataset
# * append predictions to dataset
# * fit some other model on dataset + predictions  
# 
# Algorithms used:
# * [XGBoost](https://github.com/dmlc/xgboost)
# * [StackNet](https://github.com/kaz-Anova/StackNet)
# * [Extra Trees](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html)

# ![workflow](https://raw.githubusercontent.com/vecxoz/kag17_2sigma_renthop/master/workflow.png)

# # Solution

# ## Import

# In[1]:

# Basics
import os
import sys
import gc
import re
from subprocess import check_output

# Math stack
import numpy as np
np.set_printoptions(suppress = True)
import pandas as pd
# pd.set_option('display.float_format', lambda x: '%.6f' % x)
pd.options.mode.chained_assignment = None  # default = 'warn'
from scipy import sparse
from scipy.optimize import minimize

# Preprocessing and scoring
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold, train_test_split
from sklearn.metrics import log_loss, make_scorer
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Text (vectorizing, stamming, sentiment)
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
# from textblob import TextBlob

# Models
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.ensemble import ExtraTreesClassifier

# Default value to fill NaN
fill_val = 0


# ## Load data

# In[2]:

data_dir = './data/'
train_df = pd.read_json(data_dir + 'train.json')
test_df = pd.read_json(data_dir + 'test.json')
subm_df = pd.read_csv(data_dir + 'sample_submission.csv')
# Load "magic feature"
time_df = pd.read_csv(data_dir + 'listing_image_time.csv')
# Rename columns
time_df.columns = ['listing_id', 'timestamp']
#
print(train_df.shape) # (49352, 15)
print(test_df.shape) # (74659, 14)


# ## Combine train and test to simplify feature calculation

# In[3]:

y_col = 'interest_level'
r, c = train_df.shape
test_df.loc[:, y_col] = 'na'
tt_df = pd.concat([train_df, test_df], ignore_index = True)
# Merge with "magic feature"
tt_df = pd.merge(tt_df, time_df, on = 'listing_id', how = 'left')
print(tt_df.shape) # (124011, 16)


# ## Minimal outlier correction

# In[4]:

# bathrooms
tt_df.loc[69023, 'bathrooms'] = 2 # was 112
tt_df.loc[72329, 'bathrooms'] = 2 # was 20
tt_df.loc[113071, 'bathrooms'] = 2 # was 20
tt_df.loc[1990, 'bathrooms'] = 1 # was 10

# lat/lon - just another city - e.g. LA - leave as is

# price
tt_df.loc[25538, 'price'] = 1025 # was 111111 # real number from dscription
tt_df.loc[tt_df['price'] > 100000, 'price'] = 100000 # low interest_level for all

# timestamp
tt_df.loc[35264, 'timestamp'] = 1479787252 # was 1491289977 (only one record from april) # replace with last timestamp excluding this record


# ## Feature engineering

# In[5]:

#---------------------------------------------------------------------------
# Features denoting presence of NaNs, zeros, outliers
#---------------------------------------------------------------------------
tt_df['building_id_is_zero'] = (tt_df['building_id'].apply(len) == 1).astype(np.int64)
#---------------------------------------------------------------------------
# Count/len
#---------------------------------------------------------------------------
tt_df['num_photos'] = tt_df['photos'].apply(len) # number of photos
tt_df['num_features'] = tt_df['features'].apply(len) # number of 'features'
tt_df['num_description_words'] = tt_df['description'].apply( lambda x: len(x.split(' ')) ) # number of words in description
#---------------------------------------------------------------------------
# Date/Time
#---------------------------------------------------------------------------
tt_df['created'] = pd.to_datetime(tt_df['created']) # convert the created column to datetime 
# tt_df['year'] = tt_df['created'].dt.year # year is constant for this dataset
tt_df['month'] = tt_df['created'].dt.month
tt_df['day'] = tt_df['created'].dt.day
tt_df['hour'] = tt_df['created'].dt.hour
#---------------------------------------------------------------------------
# Rooms
#---------------------------------------------------------------------------
tt_df['bad_plus_bath'] = tt_df['bedrooms'] + tt_df['bathrooms']
tt_df['more_bed'] = (tt_df['bedrooms'] > tt_df['bathrooms']).astype(np.int64)
tt_df['more_bath'] = (tt_df['bedrooms'] < tt_df['bathrooms']).astype(np.int64)
tt_df['bed_bath_equal'] = (tt_df['bedrooms'] == tt_df['bathrooms']).astype(np.int64)
tt_df['bed_bath_diff'] = tt_df['bedrooms'] - tt_df['bathrooms']
tt_df['bed_bath_ration'] = tt_df['bedrooms'] / tt_df['bathrooms']
tt_df['bed_bath_ration'] = tt_df['bed_bath_ration'].replace([np.inf], np.max(tt_df.loc[tt_df['bed_bath_ration'] != np.inf, 'bed_bath_ration']) + 1)
tt_df['bed_bath_ration'].fillna(0, inplace = True)

tt_df['bath_is_int'] = (0 == tt_df['bathrooms'] % 1).astype(np.int64)
# tt_df['diff_rooms_photos'] = tt_df['num_photos'] - tt_df['bad_plus_bath']
tt_df['bed_bath_photos_ration'] = tt_df['bad_plus_bath'] / tt_df['num_photos']
tt_df['bed_bath_photos_ration'] = tt_df['bed_bath_photos_ration'].replace([np.inf], np.max(tt_df.loc[tt_df['bed_bath_photos_ration'] != np.inf, 'bed_bath_photos_ration']) + 1)
tt_df['bed_bath_photos_ration'] = tt_df['bed_bath_photos_ration'].replace([np.nan], 0)
#---------------------------------------------------------------------------
# Price
#---------------------------------------------------------------------------
tt_df['price_per_bed'] = tt_df['price'] / tt_df['bedrooms']
tt_df['price_per_bed'] = tt_df['price_per_bed'].replace([np.inf], np.max(tt_df.loc[tt_df['price_per_bed'] != np.inf, 'price_per_bed']) + 1)
tt_df['price_per_bath'] = tt_df['price'] / tt_df['bathrooms']
tt_df['price_per_bath'] = tt_df['price_per_bath'].replace([np.inf], np.max(tt_df.loc[tt_df['price_per_bath'] != np.inf, 'price_per_bath']) + 1)
tt_df['price_per_bed_plus_bath'] = tt_df['price'] / (tt_df['bedrooms'] + tt_df['bathrooms'])
tt_df['price_per_bed_plus_bath'] = tt_df['price_per_bed_plus_bath'].replace([np.inf], np.max(tt_df.loc[tt_df['price_per_bed_plus_bath'] != np.inf, 'price_per_bed_plus_bath']) + 1)
tt_df['price_per_photo'] = tt_df['price'] / tt_df['num_photos']
tt_df['price_per_photo'] = tt_df['price_per_photo'].replace([np.inf], np.max(tt_df.loc[tt_df['price_per_photo'] != np.inf, 'price_per_photo']) + 1)
#---------------------------------------------------------------------------
# Address (case may contain info)
#---------------------------------------------------------------------------
tt_df['street_address'] = tt_df['street_address'].apply(lambda x: x.lower())
tt_df['display_address'] = tt_df['display_address'].apply(lambda x: x.lower())
# tt_df['disp_addr_is_not_in_street_addr'] = tt_df[['street_address', 'display_address']].apply(lambda x: np.int(-1 == x.street_address.find(x.display_address)), axis = 1)
#---------------------------------------------------------------------------
# Lat/Lon
#---------------------------------------------------------------------------
# # latlon count (density of points)
tt_df['latlon'] = tt_df['longitude'].round(3).astype(str) + '_' + tt_df['latitude'].round(3).astype(str)

latlon_count = tt_df['latlon'].value_counts()
latlon_count = latlon_count.reset_index().rename(columns = {'index':'latlon', 'latlon':'density'})
tt_df = pd.merge(tt_df, latlon_count, on = 'latlon', how = 'left')

# Distance to New-Yourk center
center_lat = 40.785091
center_lon = -73.968285
tt_df['euclid_dist_to_center'] = np.sqrt((tt_df['latitude'] - center_lon) ** 2  + (tt_df['longitude'] - center_lat) ** 2)
    
# Rotation for different angles
for angle in [15,30,45,60]:
    namex = 'rot' + str(angle) + '_x'
    namey = 'rot' + str(angle) + '_y'
    alpha = np.pi / (180 / angle)
    
    tt_df[namex] = tt_df['latitude'] * np.cos(alpha) + tt_df['longitude'] * np.sin(alpha)
    tt_df[namey] = tt_df['longitude'] * np.cos(alpha) - tt_df['latitude'] * np.sin(alpha)
    
#---------------------------------------------------------------------------
# Categotical
#---------------------------------------------------------------------------
# Label encoding
categorical_cols = ['display_address', 'manager_id', 'building_id', 'street_address']
for col in categorical_cols:
    le = LabelEncoder()
    tt_df.loc[:, col] = le.fit_transform(tt_df[col].values)
    
# Manager count
man_count = tt_df['manager_id'].value_counts()
man_count = man_count.reset_index().rename(columns = {'index':'manager_id', 'manager_id':'man_count'})
tt_df = pd.merge(tt_df, man_count, on = 'manager_id', how = 'left')

# Building count
build_count = tt_df['building_id'].value_counts()
build_count = build_count.reset_index().rename(columns = {'index':'building_id', 'building_id':'build_count'})
tt_df = pd.merge(tt_df, build_count, on = 'building_id', how = 'left')

# Top5 building
build_count = tt_df['building_id'].value_counts()
p = np.percentile(build_count.values, 95)
tt_df['top_5_building'] = tt_df['building_id'].apply( lambda x: np.int(x in build_count.index.values[build_count.values >= p]) )
#---------------------------------------------------------------------------
# Dscription
# Description in fact is the list of features, so probably it can add little values to 'features'
#---------------------------------------------------------------------------
tt_df['number_of_new_lines'] = tt_df['description'].apply(lambda x: x.count('<br /><br />'))
tt_df['website_redacted'] = tt_df['description'].str.contains('website_redacted').astype(np.int)
#---------------------------------------------------------------------------
# Strange
#---------------------------------------------------------------------------
tt_df['price_is_round_sousand'] = (0 == tt_df['price'] % 1000).astype(np.int64)
tt_df['price_is_round_hundred'] = (0 == tt_df['price'] % 100).astype(np.int64)
#---------------------------------------------------------------------------
# Image timestamp ('magic feature')
#---------------------------------------------------------------------------
tt_df['ts_date'] = pd.to_datetime(tt_df['timestamp'], unit = 's')

# tt_df['ts_days_passed'] = (tt_df['ts_date'].max() - tt_df['ts_date']).astype('timedelta64[D]').astype(int)
tt_df['ts_month'] = tt_df['ts_date'].dt.month
tt_df['ts_week'] = tt_df['ts_date'].dt.week
tt_df['ts_day'] = tt_df['ts_date'].dt.day
# tt_df['ts_dayofweek'] = tt_df['ts_date'].dt.dayofweek
tt_df['ts_dayofyear'] = tt_df['ts_date'].dt.dayofyear
tt_df['ts_hour'] = tt_df['ts_date'].dt.hour
tt_df['ts_tensdays'] = tt_df['ts_day'].apply(lambda x: 1 if x < 10 else 2 if x < 20 else 3)

#---------------------------------------------------------------------------
# Check NaNs
#---------------------------------------------------------------------------
print(tt_df.shape) # (124011, 60)
print('NaN: %s' % tt_df.isnull().mean().any())


# ## Split dataset to calculate features based on taraget variable (probabilities)

# In[6]:

train_df = tt_df[:r]
test_df = tt_df[r:]


# ## Features based on taraget variable (probabilities)
# ** * Perform groupping of interest_level by (manager_id and interest_level) * **  
# ** * Should be very careful to avoid leakage. For example more than 5 folds will increase possibility of leakage * **

# ## Function to calculate probabilities

# In[7]:

def get_prob(df, col = None, agg_func = None):
    """
    Params
    ------
    df - Panadas dataframe
    col - column of interest
    agg_func - aggregation function
    
    Return
    ------
    Pandas dataframe ready to merge with df on manager_id
    
    Logic
    -----
    
    We have this:                      We want to get this:
    -------------                      --------------------
    
    interest_level manager_id          manager_id  prob_high  prob_low  prob_medium
               low        foo                 bar   0.333333       NaN     0.666667
            medium        bar                 foo   0.200000       0.4     0.400000
            medium        foo
              high        bar
            medium        foo
            medium        bar
               low        foo
              high        foo
    """
    aggregate_df = df.groupby(['manager_id', 'interest_level'])[[col]].aggregate(agg_func).rename(columns = {col: 'aggregate'}).reset_index()
    sum_df = aggregate_df.groupby(['manager_id'])[['aggregate']].sum().rename(columns = {'aggregate': 'sum'}).reset_index()
    aggregate_df = pd.merge(aggregate_df, sum_df, on = 'manager_id', how = 'left')
    aggregate_df['prob'] = aggregate_df['aggregate'] / aggregate_df['sum']
    piv_df = pd.pivot_table(aggregate_df, values='prob', columns=['interest_level'], index = 'manager_id').reset_index()
    name = col + '_' + agg_func
    piv_df.rename(columns = {'high': 'prob_high_' + name, 'low': 'prob_low_' + name, 'medium': 'prob_medium_' + name}, inplace = True)
    return piv_df


# ## Calculate probabilities

# In[8]:

# Init CV
kf = KFold(n_splits = 5, shuffle = True, random_state = 0)

# Init aggregation
col = 'interest_level'
agg_func = 'count'

# Init new columns
for i in ['prob_high_', 'prob_low_', 'prob_medium_']: # alphabetically
    train_df.loc[:, i + col + '_' + agg_func] = fill_val

# For train set
for train_index, test_index in kf.split(train_df):
    tr_df = train_df.iloc[train_index]
    te_df = train_df.iloc[test_index]

    piv_df = get_prob(tr_df, col = col, agg_func = agg_func)
    te_df = pd.merge(te_df, piv_df, on = 'manager_id', how = 'left')
    train_df.iloc[test_index, -3:] = te_df.iloc[:, -3:].values
    print('Fold done')

# For test set
piv_df = get_prob(train_df, col = col, agg_func = agg_func)
test_df = pd.merge(test_df, piv_df, on = 'manager_id', how = 'left')

# Fill NaN
train_df.fillna(fill_val, inplace = True)
test_df.fillna(fill_val, inplace = True)

print(train_df.shape) # (49352, 63)
print(test_df.shape) # (74659, 63)


# ## Encode target

# In[9]:

train_df.loc[:, 'interest_level'] = train_df['interest_level'].map({'high': 0, 'medium': 1, 'low': 2})
y_train = train_df['interest_level'].values


# ## Combine train and test to work with text features

# In[10]:

# Combine train and test
tt_df = pd.concat([train_df, test_df], ignore_index = True)
print(tt_df.shape) # (124011, 63)


# ## Create text features (sparse) from 'features' column

# In[11]:

# Text features from 'features'
tt_df['features'] = tt_df['features'].apply(lambda x: ' '.join(['_'.join(i.split(' ')) for i in x]))
vectorizer = CountVectorizer(stop_words = 'english', max_features = 200)
tt_sparse = vectorizer.fit_transform(tt_df['features'])


# ## Create text features from 'description' column (sentiment, NO improvement)

# In[12]:

# Text features from 'description'
# tt_df['sentiment_polarity'] = tt_df['description'].apply(lambda x: TextBlob(x).sentiment.polarity)
# tt_df['sentiment_subjectivity'] = tt_df['description'].apply(lambda x: TextBlob(x).sentiment.subjectivity)


# ## Create final dataset
# 
# * Explicitly select columns to use
# * Combine dense and sparse data into a single sparse dataset
# * Check consistency: NaN, +INF, -INF, constant columns, duplicated columns

# In[13]:

X_cols = ['bathrooms', 'bedrooms', 'latitude', 'longitude', 'price', 
          'listing_id', 'num_photos', 'num_features', 'num_description_words', 
          'month', 'day', 'hour', 'bad_plus_bath', 'more_bed', 
          'more_bath', 'bed_bath_equal', 'bed_bath_diff', 'bed_bath_ration', 
          'price_per_bed', 'price_per_bath', 'price_per_bed_plus_bath', 
          'price_per_photo', 'price_is_round_sousand', 'price_is_round_hundred', 
          'building_id_is_zero', 'bath_is_int', 'bed_bath_photos_ration', 
          'density', 'euclid_dist_to_center', 
          'prob_high_interest_level_count', 'prob_low_interest_level_count', 'prob_medium_interest_level_count', 
          'display_address', 'manager_id', 'building_id', 'street_address', 
          'man_count', 'build_count', 'top_5_building', 
          
          'ts_month', 'ts_week', 'ts_day', 'ts_dayofyear', 'ts_hour', 'ts_tensdays',
          
          'rot15_x', 'rot15_y', 'rot30_x', 'rot30_y', 
          'rot45_x', 'rot45_y', 'rot60_x', 'rot60_y',
          
          'number_of_new_lines',  'website_redacted',
          ]

TT = sparse.hstack([tt_df[X_cols], tt_sparse]).tocsr()
# TT = sparse.csr_matrix(tt_df[X_cols]) # without text features

# Check for NaN, INF, -INF
print('NaN   -> ', np.bool(np.mean(np.isnan(TT.toarray())))) # should be False
print('+INF  -> ', np.bool(np.mean(np.isinf(TT.toarray())))) # should be False
print('-INF  -> ', np.bool(np.mean(np.isneginf(TT.toarray())))) # should be False

# Check for constant fetures
print('CONST -> ', np.bool(np.mean(TT[0] == TT.mean(axis = 0)))) # should be False

# Check for duplicate entries in column (feature) list
print('DUPL  -> ', len(X_cols) != len(set(X_cols))) # should be False

# Split
X_train = TT[:r]
X_test = TT[r:]

# Shape
print('SHAPE -> ', X_train.shape, X_test.shape) # (49352, 255) (74659, 255)


# ## Organize cross-validation for manual parameter tuning and feature selection

# In[14]:

# Init model
model = XGBClassifier(seed = 0, objective = 'multi:softprob', 
                      learning_rate = 0.1, n_estimators = 100, 
                      max_depth = 6, min_child_weight = 1, 
                      subsample = 0.7, colsample_bytree = 0.7)

# Crate sklearn scorer
scorer = make_scorer(log_loss, needs_proba = True)
# Run CV and get mean score
print(np.mean(cross_val_score(model, X_train, y_train, cv = 3, scoring = scorer)))


# ## Function to create out-of-fold predictions

# In[15]:

def oof(model, X_train, y_train, X_test, oof_test = True):
    """
    Parameters
    ----------
    Self-explanatory
    oof_test - if True, then predict test set
    
    Return
    ------
    S_train - OOF predictions for train set
    S_test  - prediction for test set (fit model on full train set)
    """
    # Init CV
    kf = KFold(n_splits = 3, shuffle = True, random_state = 0)
    
    # Create empty numpy arrays for stacking features
    S_train = np.zeros((X_train.shape[0], 3))
    S_test = np.zeros((X_test.shape[0], 3))
    
    # Create oof predictions for train set
    for i, (train_index, test_index) in enumerate(kf.split(X_train)):
        X_tr = X_train[train_index]
        y_tr = y_train[train_index]
        X_te = X_train[test_index]
        y_te = y_train[test_index]

        model = model.fit(X_tr, y_tr)
        y_te_pred = model.predict_proba(X_te)
        S_train[test_index, :] = y_te_pred
        print( 'Fold %d: %.6f' % (i, log_loss(y_te, y_te_pred)) )

    # Score over full dataset (mean)
    print( 'Mean:   %.6f' % log_loss(y_train, S_train) )
    
    # Create prediction for test set (fit on full train)
    if oof_test:
        model = model.fit(X_train, y_train)
        S_test = model.predict_proba(X_test)
    
    return (S_train, S_test)


# # First-level model 1: XGBoost

# ## Determine number of rounds for XGBoost using native CV

# In[16]:

# Parameters
params = {'seed': 0,
          'objective': 'multi:softprob',
          'eval_metric': 'mlogloss',
          'num_class': 3,
          'eta': 0.02,
          'max_depth': 6,
          'min_child_weight': 1,
          'subsample': 0.7,
          'colsample_bytree': 0.7,
          'silent': 1,
}

# Convert data to DMatrices
dtrain = xgb.DMatrix(X_train, label = y_train)
dtest = xgb.DMatrix(X_test)

# Using 3-fold CV
res = xgb.cv(
    params, 
    dtrain,
    num_boost_round = 10000,
    early_stopping_rounds = 50,
    nfold = 3,
    seed = 0,
    stratified = False,
    show_stdv = True,
    verbose_eval = 100
    )

# Output result
n_part = res.shape[0]
n_full = np.int(res.shape[0] + (1/3) * res.shape[0])
print('\ncv mean + std -> [%.6f + %.6f]\nntrees -> [%d]\nntrees for full data (+1/3) -> [%d]' % (res.iloc[-1, 0], res.iloc[-1, 1], n_part, n_full))


# ## Run XGBoost

# In[17]:

# Init model
model = XGBClassifier(seed = 0, objective = 'multi:softprob', 
                      learning_rate = 0.02, n_estimators = n_part, 
                      max_depth = 6, min_child_weight = 1, 
                      subsample = 0.7, colsample_bytree = 0.7)
# Get oof
xgb_oof_train, xgb_oof_test = oof(model, X_train, y_train, X_test, oof_test = False)

# Init model for test (as we train on full train set we need more rounds)
model = XGBClassifier(seed = 0, objective = 'multi:softprob', 
                      learning_rate = 0.02, n_estimators = n_full, 
                      max_depth = 6, min_child_weight = 1, 
                      subsample = 0.7, colsample_bytree = 0.7)

# Fit model on full train
model = model.fit(X_train, y_train)

# Predict test
xgb_oof_test = model.predict_proba(X_test)

# Export to txt files
np.savetxt(data_dir + 'xgb_oof_train.csv', xgb_oof_train, delimiter = ',', fmt = '%.5f')
np.savetxt(data_dir + 'xgb_oof_test.csv', xgb_oof_test, delimiter = ',', fmt = '%.5f') 


# # First-level model 2: StackNet

# ## Prepare data for StackNet

# In[18]:

#-------------------------------------------------------------------------------
# First column in train - labels
# First column in test - dummy (indices)
#-------------------------------------------------------------------------------

# Get test index to use as first dummy column in test set for StackNet
ids = test_df['listing_id'].values

# Concat oof and predictions from best model (xgb)
TT_dense = np.c_[TT.toarray(), np.r_[xgb_oof_train, xgb_oof_test]] # (124011, 258)

# Scale
scaler = StandardScaler()
TT_dense = scaler.fit_transform(TT_dense)

# Split
X_train_dense = TT_dense[:r] # (49352, 258)
X_test_dense = TT_dense[r:] # (74659, 258)

# Append target to train
X_train_dense = np.c_[y_train, X_train_dense] # (49352, 259)
# Append id to test
X_test_dense = np.c_[ids, X_test_dense] # (74659, 259)

# Export to txt files
np.savetxt(data_dir + 'train_std.csv', X_train_dense, delimiter = ',', fmt = '%.5f')
np.savetxt(data_dir + 'test_std.csv', X_test_dense, delimiter = ',', fmt = '%.5f') 


# ## Run StackNet
# ** * We use dummy 3-level model in file `params.txt` just to get train oof from 2-level model * **

# In[19]:

# Run StackNet and get output
stacknet_log = check_output(['bash', 'run.sh']).decode(sys.stdout.encoding)
# Save output to file
with open(data_dir + 'stacknet_log.txt', 'w') as f:
    str_len = f.write(stacknet_log)


# ## Load StackNet OOF

# In[20]:

# Load StackNet oof
stacknet_oof_train = np.loadtxt('stacknet_oof2.csv', delimiter = ',')
stacknet_oof_test = np.loadtxt('stacknet_oof_test2.csv', delimiter = ',')


# # First-level model 3: Extra Trees

# ## Run Extra Trees

# In[21]:

# Inint model
model = ExtraTreesClassifier(random_state = 0, n_jobs = -1, n_estimators = 1000, 
                               criterion = 'entropy', max_depth = None)

# Get oof
et_oof_train, et_oof_test = oof(model, np.c_[X_train_dense[:, 1:], stacknet_oof_train], 
                                y_train, np.c_[X_test_dense[:, 1:], stacknet_oof_test], oof_test = True)

# Export to txt files
np.savetxt(data_dir + 'et_oof_train.csv', et_oof_train, delimiter = ',', fmt = '%.5f')
np.savetxt(data_dir + 'et_oof_test.csv', et_oof_test, delimiter = ',', fmt = '%.5f') 


# # Ensemble

# ## Look at oof scores for our models

# In[22]:

# Output oof scores
print('XGB:      %.6f' % log_loss(y_train, xgb_oof_train))
print('StackNet: %.6f' % log_loss(y_train, stacknet_oof_train))
print('ET:       %.6f' % log_loss(y_train, et_oof_train))


# ## Perform SLSQP optimization with bounds and constraints (9 parameters for each column)

# In[23]:

#-------------------------------------------------------------------------------
# One parameter for each column
#-------------------------------------------------------------------------------

def cost(params):
    y_pred = params[:3] * xgb_oof_train + params[3:6] * stacknet_oof_train + params[6:9] * et_oof_train
    return log_loss(y_train, y_pred)
    
def con1(params):
    return params[0] + params[3] + params[6] - 1
    
def con2(params):
    return params[1] + params[4] + params[7] - 1
    
def con3(params):
    return params[2] + params[5] + params[8] - 1

# params = [0.33] * 9
# print(cost(params)) # 0.511137

n = 9
init = [0.33] * n
cons = ({'type': 'eq', 'fun': con1},
        {'type': 'eq', 'fun': con2},
        {'type': 'eq', 'fun': con3})
bounds = [(0, 1)] * n
res = minimize(cost, init, method = 'SLSQP', bounds = bounds, constraints = cons, options = {'maxiter': 100000})


# ## Optimization result

# In[25]:

print(res)


# # Create submission

# In[26]:

params = res['x']
y_pred = params[:3] * xgb_oof_test + params[3:6] * stacknet_oof_test + params[6:9] * et_oof_test
        
subm_df.loc[:, 'listing_id'] = test_df['listing_id'].values
subm_df.iloc[:, 1:] = y_pred
subm_df.to_csv('./reproduced_submission/reproduced_submission.csv', index = False)


# # Conclusion

# 1. ** * Relatively simple solution with strong result * **  
# 2. ** * A lot of fun with cool dataset and competition* **
