import pandas as pd
import json
import re
import numpy as np
from sklearn.linear_model import LinearRegression

#title, year, genre, runtime, director, actor, budget, revenue,

def clean_alt_list(list_):
    list_ = list_.replace(', ', '","')
    list_ = list_.replace('[', '["')
    list_ = list_.replace(']', '"]')
    return list_



def readDataset1():
    col_list = ['title', 'year', 'genre', 'duration', 'director', 'actors', 'budget', 'worlwide_gross_income', 'country', 'language']
    ds1 = pd.read_csv('dataset1/IMDb movies.csv', usecols=col_list)

    ds1 = ds1[(ds1['country'] == 'USA') | (ds1['language'] == 'English') | (ds1['country'] == 'UK')]


    ds1.rename(columns= {'duration': 'runtime', 'actors': 'actor', 'worlwide_gross_income': 'revenue'}, inplace = True)
    # print(ds1)

    del ds1['country']
    del ds1['language']

    return ds1

def readDataset3():
    col_list = ['name', 'year', 'genre', 'runtime', 'director', 'star', 'budget', 'gross', 'country']
    ds3 = pd.read_csv('dataset3.csv', usecols=col_list, encoding='latin-1')

    ds3 = ds3[(ds3['country'] == 'USA') | (ds3['country'] == 'UK')]

    ds3.rename(columns= {'name': 'title', 'star': 'actor', 'gross': 'revenue'}, inplace = True)
    # print(ds3)

    del ds3['country']
    return ds3

def readDataset4():
    col_list = ['movie_title', 'title_year', 'genres', 'duration', 'director_name', 'actor_1_name', 'budget', 'gross', 'country', 'language']

    ds4 = pd.read_csv('dataset4.csv', usecols=col_list, encoding='latin-1')

    ds4 = ds4[(ds4['country'] == 'USA') | (ds4['language'] == 'English') | (ds4['country'] == 'UK')]

    ds4.rename(columns= {'movie_title': 'title', 'title_year': 'year', 'genres': 'genre', 'duration': 'runtime', 'director_name': 'director', 'actor_1_name': 'actor', 'gross': 'revenue'}, inplace = True)
    # print(ds4)

    del ds4['country']
    del ds4['language']

    return ds4



if __name__ == '__main__':
    ds1 = readDataset1()
    ds2 = readDataset3()
    #ds3 = readDataset4()
    # readDataset5()
    # readDataset6()

    data = pd.concat([ds1, ds2])
    

    data['year'] = pd.to_numeric(data['year'], errors = 'coerce')
    
    data = data[data['year'].notna()]

    data['year'] = data['year'].astype(int)

    data['title'] = data["title"].apply(lambda x: ''.join([" " if ord(i) < 32 or ord(i) > 126 else i for i in x]))

    data['budget'] = data['budget'].str.replace(',', '')
    data['budget'] = data['budget'].str.replace('$', '')
    data['budget'] = data['budget'].str.replace(r'[^\d.]+', '')

    data['revenue'] = data['revenue'].str.replace(',', '')
    data['revenue'] = data['revenue'].str.replace('$', '')
    data['revenue'] = data['revenue'].str.replace(r'[^\d]+', '')

    

    data['budget'] = pd.to_numeric(data['budget'], errors = 'coerce')
    data['revenue'] = pd.to_numeric(data['revenue'], errors = 'coerce')

    data['budget'].replace(np.nan, 0, inplace = True)
    data['revenue'].replace(np.nan, 0, inplace = True)

    data = data.drop_duplicates(subset = 'title')

    data = data[data['year'] > 1970]

    print(data)

    data2 = pd.isnull(data)


    dataFiltered = data[(data['budget'] != 0) | (data['revenue'] != 0)]

    # count = 0
    # for i, j in data2.iterrows():
    #     if j['budget'] == True and j['revenue'] == True:
    #         data.drop(i, inplace = True)

    dataFiltered['budget'].replace( 0, np.nan, inplace = True)
    dataFiltered['revenue'].replace(0, np.nan, inplace = True)

    dataFiltered['revenue']= dataFiltered['revenue'].fillna(dataFiltered['revenue'].median())
    dataFiltered['budget']= dataFiltered['budget'].fillna(dataFiltered['budget'].median())
    # print(count)


    print(dataFiltered.info())

   # compression_opts = dict(method='zip',archive_name = 'out.csv')  
   # dataFiltered.to_csv('out.zip', index = False,compression = compression_opts)
    
    cols = ["runtime", "budget","revenue"]
    
    df = dataFiltered[cols]
    
    test_df = df[df["budget"].isnull()]
    
    df = df.dropna()
    
    y_train = df["budget"]
    X_train = df.drop("budget", axis=1)
    X_test = test_df.drop("budget", axis=1)
    
   
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    
    dataFiltered.loc[dataFiltered.budget.isnull(), 'budget'] = y_pred
    
  
    dataFiltered['roi']=((dataFiltered['revenue']- dataFiltered['budget']) / dataFiltered['budget'])*100
    
    
    
   

    
