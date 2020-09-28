#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer


# In[ ]:


def category_onehot_multcols(multcolumns,df):
    df_final=df.copy()
    i=0
    for fields in multcolumns:
        
        #print(fields,end=",")
        df1=pd.get_dummies(df_final[fields],drop_first=True)
        
        df_final.drop([fields],axis=1,inplace=True)
        if i==0:
            dFinal=df1.copy()
        else:
            dFinal=pd.concat([dFinal,df1],axis=1)
        i=i+1
        
    dFinal= pd.concat([df_final,dFinal],axis=1)

    return dFinal



def clean_df_train(df1):
    
    print("Drop columns with more than 50% of missing values")
    print("droping ID column from dataset")
    
    #droping columns that have more than 50% of missing values
    column_nans = df1.isnull().mean()
    drop_cols = df1.columns[column_nans > 0.50]
    print('columns to drop: ', drop_cols)
    
    df1=df1.drop(['Alley','PoolQC','Fence','MiscFeature'],axis=1)
    
    print("create a copy of data frame")
    df=df1.copy()
   
    
    #Imputing null values with mean(for numerical features) and median(for categorical features)
    print("Imputing null values")
    df['LotFrontage']=df['LotFrontage'].fillna(df['LotFrontage'].mean())
    df['GarageYrBlt']=df['GarageYrBlt'].fillna(df['GarageYrBlt'].mean())
    df['MasVnrType']=df['MasVnrType'].fillna(df['MasVnrType'].mode()[0])
    df['BsmtQual']=df['BsmtQual'].fillna(df['BsmtQual'].mode()[0])
    df['BsmtCond']=df['BsmtCond'].fillna(df['BsmtCond'].mode()[0])
    df['BsmtExposure']=df['BsmtExposure'].fillna(df['BsmtExposure'].mode()[0])
    df['BsmtFinType1']=df['BsmtFinType1'].fillna(df['BsmtFinType1'].mode()[0])
    df['BsmtFinType2']=df['BsmtFinType2'].fillna(df['BsmtFinType2'].mode()[0])
    df['FireplaceQu']=df['FireplaceQu'].fillna(df['FireplaceQu'].mode()[0])
    df['GarageType']=df['GarageType'].fillna(df['GarageType'].mode()[0])
    df['GarageFinish']=df['GarageFinish'].fillna(df['GarageFinish'].mode()[0])
    df['GarageQual']=df['GarageQual'].fillna(df['GarageQual'].mode()[0])
    df['GarageCond']=df['GarageCond'].fillna(df['GarageCond'].mode()[0])
    
    
    
    print("droping null values")
    df.dropna(inplace=True)
    
    category_columns=df.select_dtypes(include=['object']).columns
    numerical_columns=df.select_dtypes(include=['number']).columns
    
    print("creating dummy columns")
    cat_df=category_onehot_multcols(category_columns,df)
    
    
    num_df=df[numerical_columns]
    final_df=pd.concat([cat_df,num_df],axis=1)
   
    print("droping duplicate columns")
    final_df=final_df.loc[:,~final_df.columns.duplicated()]
    
    
    
    return final_df  


def clean_df_test(df1):
    
    print("Drop columns with more than 50% of missing values")
    print("droping ID column from dataset")
    
    #droping columns that have more than 50% of missing values
    column_nans = df1.isnull().mean()
    drop_cols = df1.columns[column_nans > 0.50]
    print('columns to drop: ', drop_cols)
    
    df1=df1.drop(['Alley','PoolQC','Fence','MiscFeature'],axis=1)
    
    print("create a copy of data frame")
    df=df1.copy()
   
    
    #Imputing null values with mean(for numerical features) and median(for categorical features)
    print("Imputing null values")
    df['LotFrontage']=df['LotFrontage'].fillna(df['LotFrontage'].mean())
    df['GarageYrBlt']=df['GarageYrBlt'].fillna(df['GarageYrBlt'].mean())
    df['MasVnrType']=df['MasVnrType'].fillna(df['MasVnrType'].mode()[0])
    df['BsmtQual']=df['BsmtQual'].fillna(df['BsmtQual'].mode()[0])
    df['BsmtCond']=df['BsmtCond'].fillna(df['BsmtCond'].mode()[0])
    df['BsmtExposure']=df['BsmtExposure'].fillna(df['BsmtExposure'].mode()[0])
    df['BsmtFinType1']=df['BsmtFinType1'].fillna(df['BsmtFinType1'].mode()[0])
    df['BsmtFinType2']=df['BsmtFinType2'].fillna(df['BsmtFinType2'].mode()[0])
    df['FireplaceQu']=df['FireplaceQu'].fillna(df['FireplaceQu'].mode()[0])
    df['GarageType']=df['GarageType'].fillna(df['GarageType'].mode()[0])
    df['GarageFinish']=df['GarageFinish'].fillna(df['GarageFinish'].mode()[0])
    df['GarageQual']=df['GarageQual'].fillna(df['GarageQual'].mode()[0])
    df['GarageCond']=df['GarageCond'].fillna(df['GarageCond'].mode()[0])
    
    df = df.dropna(thresh=700, axis=1)
    print("filling nan values")
    df= df.fillna(method='ffill')
    
    
    category_columns=df.select_dtypes(include=['object']).columns
    numerical_columns=df.select_dtypes(include=['number']).columns
    
    print("creating dummy columns")
    cat_df=category_onehot_multcols(category_columns,df)
    
    
    num_df=df[numerical_columns]
    final_df=pd.concat([cat_df,num_df],axis=1)
   
    print("droping duplicate columns")
    final_df=final_df.loc[:,~final_df.columns.duplicated()]
    
    
    
    return final_df  
    

    

