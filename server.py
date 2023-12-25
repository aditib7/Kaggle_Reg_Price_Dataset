from flask import Flask, jsonify, request
import pandas as pd
import joblib
import numpy as np
import sklearn
import scipy
import xgboost
import mlxtend
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
from mlxtend.regressor import StackingCVRegressor
from xgboost import XGBRegressor
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy import stats 
from scipy.stats import skew 
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax

app = Flask(__name__)

@app.route("/predict", methods=['POST'])
def do_prediction():
    
    elastic_reg = joblib.load("model/elastic_reg_model.pkl")
    lasso = joblib.load("model/lasso_model.pkl")
    ridge = joblib.load("model/ridge_model.pkl")
    svr = joblib.load("model/svr_model.pkl")
    gbr = joblib.load("model/gbr_model.pkl")
    xgb = joblib.load("model/xgb_model.pkl")
    stack_gen = joblib.load("model/stack_gen_model.pkl")
    
    json = request.get_json()
    
    test_df = pd.DataFrame.from_dict(json, orient='columns')
    
    pd.set_option('display.max_rows', 500)

    pd.set_option('display.max_columns', 313)
    
    # changing the datatypes of categorical columns to 'object' data type in test dataset

    test_df['MSSubClass'] = test_df['MSSubClass'].astype('object')
    test_df['OverallQual'] = test_df['OverallCond'].astype('object')
    test_df['YearBuilt'] = test_df['YearBuilt'].astype('object')
    test_df['YearRemodAdd'] = test_df['YearRemodAdd'].astype('object')
    test_df['GarageYrBlt'] = test_df['GarageYrBlt'].astype('object')
    test_df['YrSold'] = test_df['YrSold'].astype('object')
    test_df['MoSold'] = test_df['MoSold'].astype('object')
    
    # filling missing values
    test_df['LotFrontage']=test_df['LotFrontage'].fillna(test_df['LotFrontage'].mean())
    test_df['MSZoning']=test_df.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))
    test_df['BsmtCond']=test_df['BsmtCond'].fillna(test_df['BsmtCond'].mode()[0])
    test_df['BsmtQual']=test_df['BsmtQual'].fillna(test_df['BsmtQual'].mode()[0])
    test_df['FireplaceQu']=test_df['FireplaceQu'].fillna(test_df['FireplaceQu'].mode()[0])
    test_df['GarageType']=test_df['GarageType'].fillna(test_df['GarageType'].mode()[0])
    test_df['GarageFinish']=test_df['GarageFinish'].fillna(test_df['GarageFinish'].mode()[0])
    test_df['GarageQual']=test_df['GarageQual'].fillna(test_df['GarageQual'].mode()[0])
    test_df['GarageCond']=test_df['GarageCond'].fillna(test_df['GarageCond'].mode()[0])
    test_df['MasVnrType']=test_df['MasVnrType'].fillna(test_df['MasVnrType'].mode()[0])
    test_df['MasVnrArea']=test_df['MasVnrArea'].fillna(test_df['MasVnrArea'].mode()[0])
    test_df['BsmtExposure']=test_df['BsmtExposure'].fillna(test_df['BsmtExposure'].mode()[0])
    test_df['BsmtFinType2'] = test_df['BsmtFinType2'].fillna(test_df['BsmtFinType2'].mode()[0])
    test_df['Utilities']=test_df['Utilities'].fillna(test_df['Utilities'].mode()[0])
    test_df['Exterior1st']=test_df['Exterior1st'].fillna(test_df['Exterior1st'].mode()[0])
    test_df['Exterior2nd']=test_df['Exterior2nd'].fillna(test_df['Exterior2nd'].mode()[0])
    test_df['BsmtFinType1']=test_df['BsmtFinType1'].fillna(test_df['BsmtFinType1'].mode()[0])
    test_df['BsmtFinSF1']=test_df['BsmtFinSF1'].fillna(test_df['BsmtFinSF1'].mean())
    test_df['BsmtFinSF2']=test_df['BsmtFinSF2'].fillna(test_df['BsmtFinSF2'].mean())
    test_df['BsmtUnfSF']=test_df['BsmtUnfSF'].fillna(test_df['BsmtUnfSF'].mean())
    test_df['TotalBsmtSF']=test_df['TotalBsmtSF'].fillna(test_df['TotalBsmtSF'].mean())
    test_df['BsmtFullBath']=test_df['BsmtFullBath'].fillna(test_df['BsmtFullBath'].mode()[0])
    test_df['BsmtHalfBath']=test_df['BsmtHalfBath'].fillna(test_df['BsmtHalfBath'].mode()[0])
    test_df['KitchenQual']=test_df['KitchenQual'].fillna(test_df['KitchenQual'].mode()[0])
    test_df['Functional']=test_df['Functional'].fillna(test_df['Functional'].mode()[0])
    test_df['GarageCars']=test_df['GarageCars'].fillna(test_df['GarageCars'].mean())
    test_df['GarageArea']=test_df['GarageArea'].fillna(test_df['GarageArea'].mean())
    test_df['SaleType']=test_df['SaleType'].fillna(test_df['SaleType'].mode()[0])
    
    # dropping certain columns with significant null values
    test_df.drop(['Alley'],axis=1,inplace=True)
    test_df.drop(['GarageYrBlt'],axis=1,inplace=True)
    test_df.drop(['PoolQC','Fence','MiscFeature'],axis=1,inplace=True)
    
    # dropping because of lower correlation with 'SalePrice'
    test_df = test_df.drop(['Street', 'Utilities'], axis = 1)
    
    # box-cox transformation 
    numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    numerics = []
    for i in test_df.columns:
        if i != 'SalePrice':
            if test_df[i].dtype in numeric_dtypes:
                numerics.append(i)

    skew_features = test_df[numerics].apply(lambda x: skew(x)).sort_values(ascending=False)

    high_skew = skew_features[skew_features > 0.5]
    skew_index = high_skew.index

    for j in skew_index:
        test_df[j] = boxcox1p(test_df[j], boxcox_normmax(test_df[j] + 1))
        
        
    # Feature Transformation
    test_df['YrBltAndRemod']= test_df['YearBuilt'].apply(str) + ',' + test_df['YearRemodAdd'].apply(str)

    test_df['TotalSF']= test_df['TotalBsmtSF'] + test_df['1stFlrSF'] + test_df['2ndFlrSF']

    test_df['Total_sqr_footage'] = (test_df['BsmtFinSF1'] + test_df['BsmtFinSF2'] + test_df['1stFlrSF'] + test_df['2ndFlrSF'])

    test_df['Total_Bathrooms'] = (test_df['FullBath'] + (0.5 * test_df['HalfBath']) + test_df['BsmtFullBath'] + 
                               (0.5 * test_df['BsmtHalfBath']))

    test_df['Total_porch_sf'] = (test_df['OpenPorchSF'] + test_df['3SsnPorch'] + 
                                 test_df['EnclosedPorch'] + test_df['ScreenPorch'] + 
                                 test_df['WoodDeckSF'])
    
    
    test_df['haspool'] = test_df['PoolArea'].apply(lambda x: 1 if x > 0 else 0)

    test_df['has2ndfloor'] = test_df['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)

    test_df['hasgarage'] = test_df['GarageArea'].apply(lambda x: 1 if x > 0 else 0)

    test_df['hasbsmt'] = test_df['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)

    test_df['hasfireplace'] = test_df['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)
    
    # removing the original features after their transformation
    
    test_df = test_df.drop(['YearBuilt', 'YearRemodAdd', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'BsmtFinSF1', 'BsmtFinSF2',  
                              'FullBath', 'HalfBath', 'BsmtFullBath', 'BsmtHalfBath', 'OpenPorchSF', '3SsnPorch', 'EnclosedPorch', 'ScreenPorch', 
                              'WoodDeckSF', 'PoolArea', 'GarageArea', 'Fireplaces'], axis = 1)
    
    # creating dummy variables of categorical features

    test_df_dum = pd.get_dummies(test_df)

    # removing duplicate columns

    test_df_dum = test_df_dum.loc[:,~test_df_dum.columns.duplicated()]
    
    dff_t = test_df_dum.reset_index(inplace = True)
    
    dff_t = dff_t.drop(['Id'], axis = 1)
    
    rmv = []
    for i in dff_t.columns:
        counts = dff_t[i].value_counts()
        zeros = counts.iloc[0]
    if zeros / len(X) * 100 > 99.94:
        rmv.append(i)

        rmv = list(rmv)
        rmv.append('MSZoning_C (all)')
        
    feat_unk = dff_t.drop(rmv, axis=1).copy(deep = True)
    
    def blend_models_predict(feat_unk, elastic_reg, lasso, ridge, svr, gbr, xgb, stack_gen):
    
    return ((0.1 * elastic_reg.predict(feat_unk)) + \
            (0.05 * lasso.predict(feat_unk)) + \
            (0.1 * ridge.predict(feat_unk)) + \
            (0.1 * svr.predict(feat_unk)) + \
            (0.1 * gbr.predict(feat_unk)) + \
            (0.15 * xgb.predict(feat_unk)) + \
            (0.3 * stack_gen.predict(np.array(feat_unk))))

    print('Blending the Models and Making Predictions on the Testing Dataset:')
    print('\n')
    
    y_predict = blend_models_predict(feat_unk, elastic_reg, lasso, ridge, svr, gbr, xgb, lgb, stack_gen)

    result = {"Predicted House Price" : y_predict}
    return jsonify(result)

if __name__ == "__main__":
    app.run(host='0.0.0.0')
    
    


    


    
    

