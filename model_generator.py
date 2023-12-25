#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os


# In[2]:


get_ipython().system('pip install xgboost')


# In[3]:


get_ipython().system('pip install lightgbm')


# In[4]:


get_ipython().system('pip install mlxtend')


# In[5]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import xgboost
import lightgbm
import scipy
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
from mlxtend.regressor import StackingCVRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy import stats 
from scipy.stats import skew 
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax
from sklearn.model_selection  import train_test_split


# In[6]:


# reading training data and making the first column i.e.; 'Id' column as index column

df=pd.read_csv('https://raw.githubusercontent.com/aditib7/Kaggle_Reg_Price_Dataset/main/train%20(1).csv', delimiter = ',', index_col = 0)


# In[7]:


pd.set_option('display.max_rows', 500)

pd.set_option('display.max_columns', 313)


# In[8]:


# reading the first five rows from dataframe

df.head()


# In[9]:


#  checking the number of rows and columns in dataframe

df.shape


# In[10]:


# checking the proportion of null values for every column in dataset

null_count = df.isnull().sum()/(df.shape[0])

null_count = null_count.round(decimals = 2)

null_count


# In[11]:


# making a heatmap of null values across columns

plt.figure(figsize=(20,10))
sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='viridis')
plt.title("Missing Values", fontsize=12)
plt.xlabel('Columns', fontsize = 10) 
plt.ylabel('Missing Values', fontsize = 10)
plt.show()


# In[12]:


# checking the information about the data types of columns 

df.info()


# In[13]:


df.columns


# In[14]:


# changing the datatypes of categorical columns to 'object' data type in training data

df['MSSubClass'] = df['MSSubClass'].astype('object')
df['OverallQual'] = df['OverallCond'].astype('object')
df['YearBuilt'] = df['YearBuilt'].astype('object')
df['YearRemodAdd'] = df['YearRemodAdd'].astype('object')
df['GarageYrBlt'] = df['GarageYrBlt'].astype('object')
df['YrSold'] = df['YrSold'].astype('object')
df['MoSold'] = df['MoSold'].astype('object')


# In[15]:


# checking the statistical summary of training data

df.describe()


# In[16]:


# Fill missing values in 'LotFrontage' column with mean value of the column

df['LotFrontage']=df['LotFrontage'].fillna(df['LotFrontage'].mean())
df['BsmtCond']=df['BsmtCond'].fillna(df['BsmtCond'].mode()[0])
df['BsmtQual']=df['BsmtQual'].fillna(df['BsmtQual'].mode()[0])
df['FireplaceQu']=df['FireplaceQu'].fillna(df['FireplaceQu'].mode()[0])
df['GarageType']=df['GarageType'].fillna(df['GarageType'].mode()[0])
df['GarageFinish']=df['GarageFinish'].fillna(df['GarageFinish'].mode()[0])
df['GarageQual']=df['GarageQual'].fillna(df['GarageQual'].mode()[0])
df['GarageCond']=df['GarageCond'].fillna(df['GarageCond'].mode()[0])
df['MasVnrType']=df['MasVnrType'].fillna(df['MasVnrType'].mode()[0])
df['MasVnrArea']=df['MasVnrArea'].fillna(df['MasVnrArea'].mode()[0])
df['BsmtExposure']=df['BsmtExposure'].fillna(df['BsmtExposure'].mode()[0])
df['BsmtFinType1']=df['BsmtFinType1'].fillna(df['BsmtFinType1'].mode()[0])
df['BsmtFinType2']=df['BsmtFinType2'].fillna(df['BsmtFinType2'].mode()[0])


# In[17]:


# dropping the 'Alley' column because of null alues

df.drop(['Alley'],axis=1,inplace=True)
df.drop(['GarageYrBlt'],axis=1,inplace=True)
df.drop(['PoolQC','Fence','MiscFeature'],axis=1,inplace=True)


# In[18]:


df.shape


# In[19]:


df.isnull().sum()


# In[20]:


plt.figure(figsize=(20,10))
sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='coolwarm')
plt.title("Missing Values", fontsize=12)
plt.xlabel('Columns', fontsize = 10) 
plt.ylabel('Missing Values', fontsize = 10)
plt.show()



# In[21]:


# dropping rows containing missing values such that one found in 'Electrical' column in dataset
df.dropna(inplace=True)


# In[22]:


df.shape


# In[23]:


df.head()


# In[24]:


# checking the correlation between numeric features and target variable, 'SalePrice'

plt.figure(figsize=(20,20))
sns.heatmap(df.select_dtypes(include=[np.number]).corr(), annot = True)
plt.show()


# In[25]:


# making a copy of training data to encode categorical data and check Spearman Rank Correlation between the variables
# by first ordering the categorical data

quant = [f for f in df.select_dtypes(include = [np.number]).columns if f != 'SalePrice']
categ = df.select_dtypes(exclude = [np.number]).columns

df1 = df.copy(deep = True)

def encode(frame, feature):
    ordering = pd.DataFrame()
    ordering['val'] = frame[feature].unique()
    ordering.index = ordering.val
    ordering['spmean'] = frame[[feature, 'SalePrice']].groupby(feature).mean()['SalePrice']
    ordering = ordering.sort_values('spmean')
    ordering['ordering'] = range(1, ordering.shape[0]+1)
    ordering = ordering['ordering'].to_dict()
    
    for cat, o in ordering.items():
        frame.loc[frame[feature] == cat, feature+'_E'] = o
    
categ_encoded = []
for q in categ:  
    encode(df1, q)
    categ_encoded.append(q+'_E')
print(categ_encoded)


# In[26]:


# creating a Spearman Rank Correlation plot

def spearman(frame, features):
    spr = pd.DataFrame()
    spr['feature'] = features
    spr['spearman'] = [frame[f].corr(frame['SalePrice'], 'spearman') for f in features]
    spr = spr.sort_values('spearman')
    plt.figure(figsize=(6, 0.25*len(features)))
    sns.barplot(data=spr, y='feature', x='spearman', orient='h')

features = quant + categ_encoded

spearman(df1, features)


# In[27]:


# plotting correlation plots after ranking categorical data

plt.figure(1)
corr_m = df1[quant+['SalePrice']].corr()
sns.heatmap(corr_m)
plt.figure(2)
corr_m = df1[categ_encoded+['SalePrice']].corr()
sns.heatmap(corr_m)
plt.figure(3)
corr_m = pd.DataFrame(np.zeros([len(quant)+1, len(categ_encoded)+1]), index=quant+['SalePrice'], columns=categ_encoded+['SalePrice'])
for q1 in quant+['SalePrice']:
    for q2 in categ_encoded+['SalePrice']:
        corr_m.loc[q1, q2] = df1[q1].corr(df1[q2])
sns.heatmap(corr_m)


# In[28]:


# visualizing the high-dimensional training data in two dimensions by using TSNE (T-distributed Stochastic Neighbor Embedding) and 
# using Principal Component Analysis for dimensionality reduction and then using KMeans for clustering the data

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

features = quant + categ_encoded

model = TSNE(n_components=2, random_state=0, perplexity=50)
X = df1[features].fillna(0.).values
tsne = model.fit_transform(X)

std = StandardScaler()
s = std.fit_transform(X)
pca = PCA(n_components=30, random_state = 0)
pca.fit(s)
pc = pca.transform(s)
kmeans = KMeans(n_clusters=5, random_state = 0)
kmeans.fit(pc)

fr = pd.DataFrame({'tsne1': tsne[:,0], 'tsne2': tsne[:, 1], 'cluster': kmeans.labels_})
sns.lmplot(data=fr, x='tsne1', y='tsne2', hue='cluster', fit_reg=False)
print(np.sum(pca.explained_variance_ratio_))


# In[29]:


# checking the distribution of target variable, 'SalePrice' to know whether it follows normal distribution - it is observed 
# that target variable, 'SalePrice' has right-skewed distribution and thus, it does not follow normal distribution. 
# Log transformation has to be carried out on target variable, 'SalePrice' before regression.

from scipy import stats 

target = df['SalePrice']
plt.figure(1); plt.title('Normal')
sns.distplot(target, kde = False, fit=stats.norm)
plt.figure(2); plt.title('Log Normal')
sns.distplot(target, kde=False, fit=stats.lognorm)
plt.show()


# In[30]:


# Log Transformation of target variable, 'SalePrice' - it is evident in the below distribution plot that the target variable
# is normally distributed after Log transformation.

df['SalePrice'] = np.log1p(df["SalePrice"])

sns.distplot(df['SalePrice'])


# In[31]:


df.shape


# In[32]:


# removing outliers

df = df[df.GrLivArea < 4500]
df = df[df['LotArea'] < 100000]
df = df[df['TotalBsmtSF'] < 3000]
df = df[df['1stFlrSF'] < 2500]
df = df[df['BsmtFinSF1'] < 2000]


# In[33]:


df.shape


# In[35]:


# REMOVING Label FROM TRAINING DATA

train_df = df.drop(['SalePrice'], axis = 1)


# In[36]:


# now, we will use Box Cox transformation method on non-normal independent numeric data or features to convert it into normal independent numeric data

from scipy.stats import skew 
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax

numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numerics = []
for i in train_df.columns:
    if i != 'SalePrice':
        if train_df[i].dtype in numeric_dtypes:
            numerics.append(i)

skew_features = train_df[numerics].apply(lambda x: skew(x)).sort_values(ascending=False)

high_skew = skew_features[skew_features > 0.5]
skew_index = high_skew.index

for j in skew_index:
    train_df[j] = boxcox1p(train_df[j], boxcox_normmax(train_df[j] + 1))
    


# In[37]:


# Feature Transformation

# dropping the less significant features - it was observed before in Spearman Rank correlation that columns, 'Street' and 
# 'Utilities' have less correlation with the target variable, 'SalePrice'

train_df = train_df.drop(['Street', 'Utilities'], axis = 1)

# creating new features from existing features

train_df['YrBltAndRemod']= train_df['YearBuilt'].apply(str) + ',' + train_df['YearRemodAdd'].apply(str)

train_df['TotalSF']= train_df['TotalBsmtSF'] + train_df['1stFlrSF'] + train_df['2ndFlrSF']

train_df['Total_sqr_footage'] = (train_df['BsmtFinSF1'] + train_df['BsmtFinSF2'] + train_df['1stFlrSF'] + train_df['2ndFlrSF'])

train_df['Total_Bathrooms'] = (train_df['FullBath'] + (0.5 * train_df['HalfBath']) + train_df['BsmtFullBath'] + 
                               (0.5 * train_df['BsmtHalfBath']))

train_df['Total_porch_sf'] = (train_df['OpenPorchSF'] + train_df['3SsnPorch'] + 
                              train_df['EnclosedPorch'] + train_df['ScreenPorch'] + 
                              train_df['WoodDeckSF'])



# In[38]:


train_df['haspool'] = train_df['PoolArea'].apply(lambda x: 1 if x > 0 else 0)

train_df['has2ndfloor'] = train_df['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)

train_df['hasgarage'] = train_df['GarageArea'].apply(lambda x: 1 if x > 0 else 0)

train_df['hasbsmt'] = train_df['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)

train_df['hasfireplace'] = train_df['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)


# In[39]:


train_df = train_df.drop(['YearBuilt', 'YearRemodAdd', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'BsmtFinSF1', 'BsmtFinSF2',  
                          'FullBath', 'HalfBath', 'BsmtFullBath', 'BsmtHalfBath', 'OpenPorchSF', '3SsnPorch', 'EnclosedPorch', 'ScreenPorch', 
                          'WoodDeckSF', 'PoolArea', 'GarageArea', 'Fireplaces'], axis = 1)


# In[40]:


# creating dummy variables of categorical features

train_df_dum = pd.get_dummies(train_df)

train_df_dum.head()


# In[41]:


train_df_dum.shape


# In[42]:


# removing duplicate columns

train_df_dum = train_df_dum.loc[:,~train_df_dum.columns.duplicated()]


# In[43]:


train_df_dum.shape


# In[48]:


dff = train_df_dum


# In[49]:


feature_dff = dff.drop(['Id'], axis = 1)

label_dff = df['SalePrice']

label_dff.reset_index(drop = True, inplace = True)


# In[50]:


feature_dff.shape


# In[51]:


rmv = []
for i in feature_dff.columns:
    counts = feature_dff[i].value_counts()
    zeros = counts.iloc[0]
    if zeros / len(X) * 100 > 99.94:
        rmv.append(i)

rmv = list(rmv)
rmv.append('MSZoning_C (all)')

feat_k = feature_dff.drop(rmv, axis=1).copy(deep = True)


# In[76]:


kfolds = KFold(n_splits=10, shuffle=True, random_state=42)

def rmsle(label_dff, y_pred):
    return np.sqrt(mean_squared_error(label_dff, y_pred))

def cv_rmse(model, X = feat_k, y = label_dff):
    rmse = np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=kfolds))
    return (rmse)


# In[77]:


alphas_alt = [14.5, 14.6, 14.7, 14.8, 14.9, 15, 15.1, 15.2, 15.3, 15.4, 15.5]
alphas2 = [5e-05, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008]
e_alphas = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007]
e_l1ratio = [0.8, 0.85, 0.9, 0.95, 0.99, 1]


# In[78]:


ridge = make_pipeline(RobustScaler(), RidgeCV(alphas=alphas_alt, cv=kfolds))

lasso = make_pipeline(RobustScaler(), LassoCV(max_iter=10000000, alphas=alphas2, random_state=42, cv=kfolds))

elasticnet = make_pipeline(RobustScaler(), ElasticNetCV(max_iter=10000000, alphas=e_alphas, cv=kfolds, l1_ratio=e_l1ratio, random_state = 0))   

svr = make_pipeline(RobustScaler(), SVR(C= 20, epsilon= 0.008, gamma=0.0003))


# In[79]:


# GradientBoostingRegressor

gbr = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05, max_depth=4, max_features='sqrt', min_samples_leaf=15, min_samples_split=10, loss='huber', random_state =42)


# In[80]:


xgboost = XGBRegressor(learning_rate=0.01,n_estimators=3400, 
                       max_depth=3, min_child_weight=0,
                       gamma=0, subsample=0.7,
                       colsample_bytree=0.7,
                       objective='reg:linear', nthread=-1,
                       scale_pos_weight=1, seed=27,
                       reg_alpha=0.00006, random_state = 42)


# In[81]:


stack_gen = StackingCVRegressor(regressors=(ridge, lasso, elasticnet, gbr, xgboost, lightgbm),
                                meta_regressor=xgboost,
                                use_features_in_secondary=True, random_state = 42)


# In[82]:


print('Getting Cross Validation RMSE Scores (Mean and Standard Deviation):')
print('\n')
score = cv_rmse(ridge)
print("RIDGE: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = cv_rmse(lasso)
print("LASSO: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = cv_rmse(elasticnet)
print("elastic net: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = cv_rmse(svr)
print("SVR: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = cv_rmse(gbr)
print("gbr: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = cv_rmse(xgboost)
print("xgboost: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


# In[83]:


print('START Fit')
print('\n')
print('stack_gen')
stack_gen_model = stack_gen.fit(np.array(feat_k), np.array(label_dff))

print('elasticnet')
elastic_model_full_data = elasticnet.fit(feat_k, label_dff)

print('Lasso')
lasso_model_full_data = lasso.fit(feat_k, label_dff)

print('Ridge')
ridge_model_full_data = ridge.fit(feat_k, label_dff)

print('Svr')
svr_model_full_data = svr.fit(feat_k, label_dff)

print('GradientBoosting')
gbr_model_full_data = gbr.fit(feat_k, label_dff)

print('xgboost')
xgb_model_full_data = xgboost.fit(feat_k, label_dff)



# In[84]:


def blend_models_predict(feat_k, elastic_model_full_data, lasso_model_full_data, ridge_model_full_data, svr_model_full_data, gbr_model_full_data, 
                         xgb_model_full_data, stack_gen_model):
    
    return ((0.1 * elastic_model_full_data.predict(feat_k)) + \
            (0.05 * lasso_model_full_data.predict(feat_k)) + \
            (0.1 * ridge_model_full_data.predict(feat_k)) + \
            (0.1 * svr_model_full_data.predict(feat_k)) + \
            (0.1 * gbr_model_full_data.predict(feat_k)) + \
            (0.15 * xgb_model_full_data.predict(feat_k)) + \
            (0.3 * stack_gen_model.predict(np.array(feat_k))))


# In[86]:


print('Blending the Models and Making Predictions on Training Dataset:')
print('\n')
print('Obtaining RMSE scores on Training data:')
print(rmsle(label_dff, blend_models_predict(feat_k, elastic_model_full_data, lasso_model_full_data, ridge_model_full_data, svr_model_full_data, gbr_model_full_data, 
                                            xgb_model_full_data, stack_gen_model)))


# In[ ]:


#Saving the machine learning models to a file

joblib.dump(elastic_model_full_data, "model/elastic_reg_model.pkl")
joblib.dump(lasso_model_full_data, "model/lasso_model.pkl")
joblib.dump(ridge_model_full_data, "model/ridge_model.pkl")
joblib.dump(svr_model_full_data, "model/svr_model.pkl")
joblib.dump(gbr_model_full_data, "model/gbr_model.pkl")
joblib.dump(xgb_model_full_data, "model/xgb_model.pkl")
joblib.dump(stack_gen_model, "model/stack_gen_model.pkl")


