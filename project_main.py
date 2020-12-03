import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

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
    
#    df = data.copy()
#    
#    
#    df = df.sort_values(by=['year'])
    
#    df['year'].value_counts().sort_index().plot(kind='line')
    

    
    data = data[data['year'] > 1950]

    data2 = pd.isnull(data)


    dataFiltered = data[(data['budget'] != 0) | (data['revenue'] != 0)]
    dataFiltered = dataFiltered[(dataFiltered['budget'] > 1000000) | (dataFiltered['budget'] == 0)]
    dataFiltered = dataFiltered[(dataFiltered['revenue'] > 1000000) | (dataFiltered['revenue'] == 0)]

    dataFiltered['budget'].replace( 0, np.nan, inplace = True)
    dataFiltered['revenue'].replace(0, np.nan, inplace = True)
    

    #print(dataFiltered.corr())

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


    #print(dataFiltered.info())

    #dataFiltered.to_csv('out.csv', index = False)


#*************************Genres***************************************
    
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
        
#    plt.figure(figsize=(20, 10))
#    plt.bar(genres, avg_rois)
#    plt.show()
        
        
    genre_score = []
    
    for i, j in dataFiltered.iterrows():
        movieGenres = j['genre'].split(',')
        movieGenres = [gen.strip() for gen in movieGenres]
        scores = 0
        count = 0
        for gen in movieGenres:
            if gen == 'Musical':
                roi = genre['Music'][0]
                count += 1
            else:
                roi = genre[gen][0]
                count += 1
            scores += roi
        genre_score.append(scores/(count))
    
    dataFiltered['genre_score'] = genre_score
        
#*************************Directors***************************************
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
                
# To calculate the directors with multiple movies with highest ROIs
                
    scores1 = []
    for key, value in list(directors.items()):
        if value[1] < 2:
            scores1.append(key)
            
    
            
    for key, value in directors.items():
        l = [value[0]/value[1], value[2]]
        directors[key] = l
        
    directors = dict(sorted(directors.items(), key=lambda item: item[1], reverse=True))
    
    iterator = iter(directors.items())
    
    dirs = []
    roi = []
    avg_budget = []
    
    
    for i in range(29):
        director, l = next(iterator)
        dirs.append(director)
        roi.append(l[0])
        avg_budget.append(l[1])
    
    
    
#    plt.figure(figsize=(20, 10))
#    plt.scatter(avg_budget, roi)
#    
#    for i, d in enumerate(dirs):
#        plt.annotate(d, (avg_budget[i], roi[i]))
        
    #director_score = roi/budget
    
    director_score = []
    
    for i, j in dataFiltered.iterrows():
        movieDirectors = j['director'].split(',')
        movieDirectors = [director.strip() for director in movieDirectors]
        scores = 0
        count = 0
        for director in movieDirectors:
            if director in scores1:
                score = 0
            else:
                roi = directors[director][0]
                budget = directors[director][1]/1000000
                score = roi/budget
            count += 1
            scores += score
        director_score.append(scores/count)
    
    dataFiltered['director_score'] = director_score


#*************************Actors***************************************  
    
    dataFiltered = dataFiltered.drop(79861)
    dataFiltered = dataFiltered.drop(82493)  
    
  
    
    actors = {}
    for i, j in dataFiltered.iterrows():
        movieActors = j['actor'].split(',')
        movieActors = [actor.strip() for actor in movieActors]
        actor = movieActors[0]
        if actor in actors:
            actors[actor][0] += j['roi']
            actors[actor][1] += 1
            actors[actor][2] += j['budget']
        else:
            actors[actor] = []
            actors[actor].append(j['roi'])
            actors[actor].append(1)
            actors[actor].append(j['budget'])
            
    # To calculate the actors with multiple movies with highest ROIs
        
    scores10 = []
    for key, value in list(actors.items()):
        if value[1] < 10:
            scores10.append(key)
            
    for key, value in actors.items():
        l = [value[0]/value[1], value[2]]
        actors[key] = l
        
    actors = dict(sorted(actors.items(), key=lambda item: item[1], reverse=True))
    
    iterator = iter(actors.items())
    
    acts = []
    roi = []
    avg_budget = []
    
    
    for i in range(29):
        actor, l = next(iterator)
        acts.append(actor)
        roi.append(l[0])
        avg_budget.append(l[1])

#    plt.figure(figsize=(20, 10))
#    plt.scatter(avg_budget, roi)
#    
#    for i, d in enumerate(acts):
#        plt.annotate(d, (avg_budget[i], roi[i]))
    
    actor_scores = []
    
    for i, j in dataFiltered.iterrows():
        movieActors = j['actor'].split(',')
        movieActors = [actor.strip() for actor in movieActors]
        actor = movieActors[0]
        if actor in scores10:
            score = 0
        else:
            roi = actors[actor][0]
            budget = actors[actor][1]/1000000
            score = roi/budget
            
        actor_scores.append(score)
    dataFiltered['actor_score'] = actor_scores

#*************************ratings***************************************
    
#    ratings = {}
#    for i, j in dataFiltered.iterrows():
#        r = j['ratings']
#        if r in ratings:
#            ratings[r][0] += j['roi']
#            ratings[r][1] += 1
#        else:
#            ratings[r] = []
#            ratings[r].append(j['roi'])
#            ratings[r].append(1)
#           
#        rating = []
#        r_roi = []
#        for k,v in ratings.items():
#            rating.append(k)
#            r_roi.append(v[0]/v[1])
#        
#        
#        plt.figure(figsize=(20, 10))
#        plt.scatter(rating, r_roi)
        
    
    #*************************Final Dataset for model***************************************  
    
    ds = dataFiltered[['ratings','budget','revenue','roi','genre_score','director_score',
                   'actor_score','runtime','num_votes','year']]
    
    #Split roi into groups 0-1 -> bad(0) & greater than 1- 3 -> OK(1) & greater than 3-8->Good(2) 
    #& greater than 8 - 13 ->Excellent(3)
    roi_class = []
    for i, j in ds.iterrows():
        if j['roi'] > 0 and j['roi'] <= 1:
            roi_class.append(0)
        elif j['roi'] > 1 and j['roi'] <= 3:
            roi_class.append(1)
        elif j['roi']>3 and j['roi']<= 8:
            roi_class.append(2)
        else:
            roi_class.append(3)
            
    ds['roi_class'] = roi_class
    del ds['roi']
    #del ds['budget']
    del ds['revenue']
    
    X = ds.iloc[:, :-1].values
    y = ds.iloc[:, -1].values

    # Splitting the dataset into the Training set and Test set
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
    
    test = X_test
    
    # Feature Scaling
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

   # Training the Decision Tree Classification model on the Training set**************
#    from sklearn.tree import DecisionTreeClassifier
#    classifier = DecisionTreeClassifier(criterion = 'gini', random_state = 0)
#    classifier.fit(X_train, y_train)
#    
#    y_pred = classifier.predict(X_test)
    
    # Training the Random Forest Classification model on the Training set*******************
    from sklearn.ensemble import RandomForestClassifier
    classifier = RandomForestClassifier(n_estimators = 10, criterion = 'gini', random_state = 0)
    classifier.fit(X_train, y_train)
    
    
    
    # Training the SVM model on the Training set******************************
#    from sklearn.svm import SVC
#    classifier = SVC(kernel = 'linear', random_state = 0, decision_function_shape='ovo')
#    classifier.fit(X_train, y_train)
    
    # Training the K-NN model on the Training set*******************************
#    from sklearn.neighbors import KNeighborsClassifier
#    classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
#    classifier.fit(X_train, y_train)
    
    # Training the Naive Bayes model on the Training set
#    from sklearn.naive_bayes import GaussianNB
#    classifier = GaussianNB()
#    classifier.fit(X_train, y_train)    
    
    y_pred = classifier.predict(X_test)

   #Neural Network MLP
#    from sklearn.neural_network import MLPClassifier
#    
#    NN = MLPClassifier(random_state = 1)
#    NN.fit(X_train, y_train)
#    
#    y_pred = NN.predict(X_test)
    
    # Making the Confusion Matrix
    from sklearn.metrics import confusion_matrix, accuracy_score
    cm = confusion_matrix(y_test, y_pred)
    accuracy_score(y_test, y_pred)
    
    from sklearn import metrics
    print(metrics.classification_report(y_test,y_pred))
    
    f = ds.columns.tolist()
    f.pop(len(f)-1)
    imp_features = pd.DataFrame({'Importance':classifier.feature_importances_, 'Features': f })
    print(imp_features)
             
    
    
    
    
    