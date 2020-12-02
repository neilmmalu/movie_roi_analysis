import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

#id, title, year, genre, runtime, director, actor, budget, revenue, ratings, num_votes

def clean_alt_list(list_):
    list_ = list_.replace(', ', '","')
    list_ = list_.replace('[', '["')
    list_ = list_.replace(']', '"]')
    return list_



def readDataset1():
    col_list = ['imdb_title_id','title', 'year', 'genre', 'duration', 'director', 'actors', 'budget', 'worlwide_gross_income', 'country', 'language', 'avg_vote']
    ds1 = pd.read_csv('dataset1/IMDb movies.csv', usecols=col_list)

    ds2 = pd.read_csv('dataset1/IMDb ratings.csv', usecols=["total_votes"])

    ds1['total_votes'] = ds2['total_votes']
    print(ds1.info())
    print(ds2.info())

    ds1 = ds1[(ds1['country'] == 'USA') | (ds1['language'] == 'English') | (ds1['country'] == 'UK')]

    
    ds1.rename(columns= {'imdb_title_id': 'id','duration': 'runtime', 'actors': 'actor', 'worlwide_gross_income': 'revenue', 'avg_vote': 'ratings', 'total_votes': 'num_votes'}, inplace = True)
    # print(ds1)

    del ds1['country']
    del ds1['language']

    return ds1


def readDataset4():
    col_list = ['movie_imdb_link','movie_title', 'title_year', 'genres', 'duration', 'director_name', 'actor_1_name', 'budget', 'gross', 'country', 'language', 'imdb_score', 'num_voted_users']

    ds4 = pd.read_csv('dataset4.csv', usecols=col_list, encoding='latin-1')

    ds4 = ds4[(ds4['country'] == 'USA') | (ds4['language'] == 'English') | (ds4['country'] == 'UK')]

    ds4['movie_imdb_link'] = ds4['movie_imdb_link'].str.split('/').str[4]

    ds4.rename(columns= {'movie_imdb_link': 'id','movie_title': 'title', 'title_year': 'year', 'genres': 'genre', 'duration': 'runtime', 'director_name': 'director', 'actor_1_name': 'actor', 'gross': 'revenue', 'imdb_score': 'ratings', 'num_voted_users': 'num_votes'}, inplace = True)
    # print(ds4)

    del ds4['country']
    del ds4['language']

    return ds4



if __name__ == '__main__':
    ds1 = readDataset1()
    # ds2 = readDataset3()
    ds3 = readDataset4()
    # readDataset5()
    # readDataset6()

    data = pd.concat([ds1, ds3])
    

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

    data = data.drop_duplicates(subset = 'id')
    
    df = data.copy()
    
    import matplotlib.pyplot as plt
    
    df = df.sort_values(by=['year'])
    
    df['year'].value_counts().sort_index().plot(kind='line')
    

    
    data = data[data['year'] > 1930]

    data2 = pd.isnull(data)


    dataFiltered = data[(data['budget'] != 0) | (data['revenue'] != 0)]
    dataFiltered = dataFiltered[(dataFiltered['budget'] > 1000000) | (dataFiltered['budget'] == 0)]
    dataFiltered = dataFiltered[(dataFiltered['revenue'] > 1000000) | (dataFiltered['revenue'] == 0)]

    dataFiltered['budget'].replace( 0, np.nan, inplace = True)
    dataFiltered['revenue'].replace(0, np.nan, inplace = True)
    
    

    print(dataFiltered.corr())

    dataFiltered['revenue']= dataFiltered['revenue'].fillna(dataFiltered['revenue'].median())
    # print(count)

    
    cols = ["runtime", "budget", "revenue", "num_votes"]
    
    df = dataFiltered[cols]
    
    test_df = df[df["budget"].isnull()]
    
    df = df.dropna()
    
    y_train = df["budget"]
    X_train = df.drop("budget", axis=1)
    X_test = test_df.drop("budget", axis=1)
    
   
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)

    y_pred_rounded = [round(num) for num in y_pred]
    
    dataFiltered.loc[dataFiltered.budget.isnull(), 'budget'] = y_pred_rounded
    
  
    dataFiltered['roi']= dataFiltered['revenue']/dataFiltered['budget']
    
    
    y = dataFiltered['roi']
    dataFiltered = dataFiltered[y.between(y.quantile(.05), y.quantile(.95))]


    print(dataFiltered.info())

    #dataFiltered.to_csv('out.csv', index = False)
    
    genre = {}
    for i, j in dataFiltered.iterrows():
        movieGenres = j['genre'].split(',')
        movieGenres = [gen.strip() for gen in movieGenres]
#        print(movieGenres)
        for gen in movieGenres:
            if gen in genre:
                genre[gen][0] += j['roi']
                genre[gen][1] += 1
            else:
                genre[gen] = []
                genre[gen].append(j['roi'])
                genre[gen].append(1)
        
    genre['Music'][0] += genre['Musical'][0]
    genre['Music'][1] += genre['Musical'][1]
    
    genre.pop('Musical')
    
    genres = []
    avg_rois = []
    for key, value in genre.items():
        genres.append(key)
        avg_rois.append(value[0]/value[1])
        
    plt.figure(figsize=(20, 10))
    plt.bar(genres, avg_rois)
    plt.show()

    directors = {}
    for i, j in dataFiltered.iterrows():
        movieDirectors = j['director'].split(',')
        movieDirectors = [director.strip() for director in movieDirectors]
#        print(movieGenres)
        for director in movieDirectors:
            if director in directors:
                directors[director][0] += j['roi']
                directors[director][1] += 1
                directors[director][2] += j['budget']
            else:
                directors[director] = []
                directors[director].append(j['roi'])
                directors[director].append(1)
                directors[director].append(j['budget'])
                
    
    for key, value in list(directors.items()):
        if value[1] < 2:
            del directors[key]
            
    for key, value in directors.items():
        l = [value[0]/value[1], value[2]]
        directors[key] = l
        
    directors = dict(sorted(directors.items(), key=lambda item: item[1], reverse=True))
    
    iterator = iter(directors.items())
    
    for i in range(29):
        print(next(iterator))
