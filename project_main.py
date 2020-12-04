#Import all libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics

#id, title, year, genre, runtime, director, actor, budget, revenue, ratings, num_votes

def clean_alt_list(list_):
    list_ = list_.replace(', ', '","')
    list_ = list_.replace('[', '["')
    list_ = list_.replace(']', '"]')
    return list_

#reading Dataset1 - IMDb movies.csv
    
def readDataset1():
    col_list = ['imdb_title_id','title', 'year', 'genre', 'duration', 'director', 'actors', 'budget', 'worlwide_gross_income', 'country', 'language', 'avg_vote']
    ds1 = pd.read_csv('dataset1/IMDb movies.csv', usecols=col_list)

    ds2 = pd.read_csv('dataset1/IMDb ratings.csv', usecols=["total_votes"])
    
    #getting votes from a different csv file
    ds1['total_votes'] = ds2['total_votes']
    
    #picking english movies made in USA & UK 
    ds1 = ds1[(ds1['country'] == 'USA') | (ds1['language'] == 'English') | (ds1['country'] == 'UK')]

    #renaming the columns
    ds1.rename(columns= {'imdb_title_id': 'id','duration': 'runtime', 'actors': 'actor', 'worlwide_gross_income': 'revenue', 'avg_vote': 'ratings', 'total_votes': 'num_votes'}, inplace = True)


    del ds1['country']
    del ds1['language']

    return ds1

#reading Dataset4 - dataset4.csv

def readDataset4():
    col_list = ['movie_imdb_link','movie_title', 'title_year', 'genres', 'duration', 'director_name', 'actor_1_name', 'budget', 'gross', 'country', 'language', 'imdb_score', 'num_voted_users']

    ds4 = pd.read_csv('dataset4.csv', usecols=col_list, encoding='latin-1')
    
    #picking english movies made in USA & UK 
    ds4 = ds4[(ds4['country'] == 'USA') | (ds4['language'] == 'English') | (ds4['country'] == 'UK')]
    
    #getting IMDb ids from a http link
    ds4['movie_imdb_link'] = ds4['movie_imdb_link'].str.split('/').str[4]
    
    #renaming the columns
    ds4.rename(columns= {'movie_imdb_link': 'id','movie_title': 'title', 'title_year': 'year', 'genres': 'genre', 'duration': 'runtime', 'director_name': 'director', 'actor_1_name': 'actor', 'gross': 'revenue', 'imdb_score': 'ratings', 'num_voted_users': 'num_votes'}, inplace = True)

    del ds4['country']
    del ds4['language']

    return ds4



if __name__ == '__main__':
    ds1 = readDataset1()
    ds3 = readDataset4()

    #combining both the datasets
    data = pd.concat([ds1, ds3])
    
#******************** Data Preprocessing *****************************************************************
    
    
    data['year'] = pd.to_numeric(data['year'], errors = 'coerce')
    
    data = data[data['year'].notna()]

    data['year'] = data['year'].astype(int)

    data['title'] = data["title"].apply(lambda x: ''.join([" " if ord(i) < 32 or ord(i) > 126 else i for i in x]))
    
    #replacing characters like , $ and alphabets(INR, EUR etc.) with ''
    data['budget'] = data['budget'].str.replace(',', '')
    data['budget'] = data['budget'].str.replace('$', '')
    data['budget'] = data['budget'].str.replace(r'[^\d]+', '')

    data['revenue'] = data['revenue'].str.replace(',', '')
    data['revenue'] = data['revenue'].str.replace('$', '')
    data['revenue'] = data['revenue'].str.replace(r'[^\d]+', '')

    data['budget'] = pd.to_numeric(data['budget'], errors = 'coerce')
    data['revenue'] = pd.to_numeric(data['revenue'], errors = 'coerce')

    data['budget'].replace(np.nan, 0, inplace = True)
    data['revenue'].replace(np.nan, 0, inplace = True)

    #removing duplicate rows based on id
    data = data.drop_duplicates(subset = 'id')
    
#    ploting a graph - movie count for each year
#    df = data.copy()
#    df = df.sort_values(by=['year'])
#    df['year'].value_counts().sort_index().plot(kind='line')
    
    #more movies made after 1950 - picking only those movies
    data = data[data['year'] > 1950]

    #dropping rows that don't have both budget and revenue
    dataFiltered = data[(data['budget'] != 0) | (data['revenue'] != 0)]
#   dataFiltered = data[(data['budget'] != 0) & (data['revenue'] != 0)]
    
    #picking all rows with budget and revenue above 1 million
    dataFiltered = dataFiltered[(dataFiltered['budget'] > 1000000) | (dataFiltered['budget'] == 0)]
    dataFiltered = dataFiltered[(dataFiltered['revenue'] > 1000000) | (dataFiltered['revenue'] == 0)]

    #replacing 0 values with null to handle missing values
    dataFiltered['budget'].replace( 0, np.nan, inplace = True)
    dataFiltered['revenue'].replace(0, np.nan, inplace = True)
    
    #replacing missing values in revenue using median method
    dataFiltered['revenue']= dataFiltered['revenue'].fillna(dataFiltered['revenue'].median())
    
    #to find the correlation between each features - to decide which features are highly dependent on budget
    #print(dataFiltered.corr())
    
    #handling missing values in budget using Linear Regrssion
    cols = ["runtime", "budget", "revenue", "num_votes"]
    
    df = dataFiltered[cols]
    
    test_df = df[df["budget"].isnull()]
    
    df = df.dropna()
    
    y_train = df["budget"]
    X_train = df.drop("budget", axis=1)
    X_test = test_df.drop("budget", axis=1)
    
    linear_reg = LinearRegression()
    linear_reg.fit(X_train, y_train)
    y_pred = linear_reg.predict(X_test)

    #rounding the values predicted
    y_pred_rounded = [round(num) for num in y_pred]
    
    dataFiltered.loc[dataFiltered.budget.isnull(), 'budget'] = y_pred_rounded
    
#********************************* Calculate ROI - revenue/budget ********************************************************** 
    
    dataFiltered['roi']= dataFiltered['revenue']/dataFiltered['budget']

    y = dataFiltered['roi']
    dataFiltered = dataFiltered[y.between(y.quantile(.05), y.quantile(.95))]


    #print(dataFiltered.info())
    #dataFiltered.to_csv('out.csv', index = False)

#************************Finding features that increase ROI ****************************
    
#************************1. Genres********************************************************

    #sum roi for each genre     
    genre = {}
    for i, j in dataFiltered.iterrows():
        movieGenres = j['genre'].split(',')
        movieGenres = [gen.strip() for gen in movieGenres]
        for gen in movieGenres:
            if gen in genre:
                genre[gen][0] += j['roi']
                genre[gen][1] += 1
            else:
                genre[gen] = []
                genre[gen].append(j['roi'])
                genre[gen].append(1)
     
    #Music and Musical same
    genre['Music'][0] += genre['Musical'][0]
    genre['Music'][1] += genre['Musical'][1]
    
    genre.pop('Musical')
    
    #storing avg roi for all genres
    genres = []
    avg_rois = []
    for key, value in genre.items():
        genres.append(key)
        avg_rois.append(value[0]/value[1])

#Plotting the graph to check if roi is dependent on genre      
#    plt.figure(figsize=(20, 10))
#    plt.bar(genres, avg_rois)
#    plt.show()
        
    #calculate genre score   
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
        
#*************************2. Directors*******************************************************************
    
    #sum roi for each director  
    directors = {}
    for i, j in dataFiltered.iterrows():
        movieDirectors = j['director'].split(',')
        movieDirectors = [director.strip() for director in movieDirectors]
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
                

    #pick directors that have made atleast 3 movies            
    scores1 = []
    for key, value in list(directors.items()):
        if value[1] < 2:
            scores1.append(key)
            
    #storing avg roi for all directors    
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
    
    
#Ploting each director based on budget and roi   
#    plt.figure(figsize=(20, 10))
#    plt.scatter(avg_budget, roi)
#    
#    for i, d in enumerate(dirs):
#        plt.annotate(d, (avg_budget[i], roi[i]))
        

    #calculate director score
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
    
    #droping rows with no actor name
    dataFiltered = dataFiltered.drop(79861)
    dataFiltered = dataFiltered.drop(82493)  
    
    #dataFiltered = dataFiltered.drop(52416)  
  
    #find roi for each main actor
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
      
        
    #pick actors who have been in atleast 11 movies
    scores10 = []
    for key, value in list(actors.items()):
        if value[1] < 10:
            scores10.append(key)
      
    #storing avg roi for each actor
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

#Ploting each actor based on budget and roi 
#    plt.figure(figsize=(20, 10))
#    plt.scatter(avg_budget, roi)
#    
#    for i, d in enumerate(acts):
#        plt.annotate(d, (avg_budget[i], roi[i]))
    
    #calculate actor score
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

#*************************3. ratings***************************************
  
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
       
#        plt.figure(figsize=(20, 10))
#        plt.scatter(rating, r_roi)
        
    
#*************************Final Dataset for model***************************************  
    
    #picking only the columns needed for the model
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
    del ds['revenue']
    
    X = ds.iloc[:, :-1].values
    y = ds.iloc[:, -1].values

#    Spliting Dataset into training and test set - Holdout method
#    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)


    #Normalize all the values
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    #training the Random Forest Classification model using LeaveOne-out k-fold Cross validation
    
    y_true = []
    y_pred = []
    
    leave_one_out = LeaveOneOut()
    for train_index, test_index in leave_one_out.split(X):
        X_train, X_test = X[train_index, :], X[test_index, :]
        y_train, y_test = y[train_index], y[test_index]
        classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
        classifier.fit(X_train, y_train)
        y_cap = classifier.predict(X_test)
        y_true.append(y_test[0])
        y_pred.append(y_cap[0])
    
    acc = accuracy_score(y_true, y_pred)
    print(acc)
    
    
#************training the Decision Tree Classification model using training set*************************
#    from sklearn.tree import DecisionTreeClassifier
#    classifier = DecisionTreeClassifier(criterion = 'gini', random_state = 0)
#    classifier.fit(X_train, y_train)
   
  
#************training the SVM model using the training set************************************************
#    from sklearn.svm import SVC
#    classifier = SVC(kernel = 'linear', random_state = 0, decision_function_shape='ovo')
#    classifier.fit(X_train, y_train)
    
#************training the K-NN model using the training set************************************************
#    from sklearn.neighbors import KNeighborsClassifier
#    classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
#    classifier.fit(X_train, y_train)
    
#************training the Naive Bayes model using the training set ****************************************
#    from sklearn.naive_bayes import GaussianNB
#    classifier = GaussianNB()
#    classifier.fit(X_train, y_train)    
    
#************training the Neural Network MLP using training set ********************************************
#    from sklearn.neural_network import MLPClassifier   
#    classifier = MLPClassifier(random_state = 1)
#    classifier.fit(X_train, y_train)

   
#    y_pred = classifier.predict(X_test)
#    acc = accuracy_score(y_test, y_pred)
#    print(acc))
    
#    print(metrics.classification_report(y_test,y_pred))

#    f = ds.columns.tolist()
#    f.pop(len(f)-1)
#    imp_features = pd.DataFrame({'Importance':classifier.feature_importances_, 'Features': f })
#    print(imp_features)


   
    #List of top 50 actors that increase ROI
    dataFiltered.sort_values('actor_score', inplace = True, ascending = False)
    reliable_actors = dataFiltered[['actor','actor_score', 'roi', 'budget']].copy()
    reliable_actors = reliable_actors[(reliable_actors['budget'] < 15000000) & (reliable_actors['roi'] > 7)]
    rel_actors = {}
    count = 0
    for i, j in reliable_actors.iterrows():
        movieActors = j['actor'].split(',')
        movieActors = [act.strip() for act in movieActors]
        actor = movieActors[0]
        if actor in rel_actors:
            if j['roi'] > rel_actors[actor][0]:
                rel_actors[actor][0] = j['roi']
                rel_actors[actor][1] = j['budget']
        else:
            rel_actors[actor] = []
            rel_actors[actor].append(j['roi'])
            rel_actors[actor].append(j['budget'])
            count += 1
        if count == 50:
            break
        
    #List of top 50 directors that increase ROI
    dataFiltered.sort_values('director_score', inplace = True, ascending = False)
    reliable_directors = dataFiltered[['director','director_score', 'roi', 'budget']].copy()
    reliable_directors = reliable_directors[(reliable_directors['budget'] < 17000000) & (reliable_directors['roi'] > 11)]
    rel_dirs = {}
    count = 0
    for i, j in reliable_directors.iterrows():
        movieDirs = j['director'].split(',')
        movieDirs = [d.strip() for d in movieDirs]
        d = movieDirs[0]
        if d in rel_dirs:
            if j['roi'] > rel_dirs[d][0]:
                rel_dirs[d][0] = j['roi']
                rel_dirs[d][1] = j['budget']
        else:
            rel_dirs[d] = []
            rel_dirs[d].append(j['roi'])
            rel_dirs[d].append(j['budget'])
            count += 1
        if count == 50:
            break
    
    a_roi = []
    a_budget = []
    a = []
    for key, value in rel_actors.items():
        a.append(key)
        a_roi.append(value[0])
        a_budget.append(value[1])
    
    
    plt.figure(figsize=(20, 20))
    plt.scatter(a_roi, a_budget)
    
    for i, d in enumerate(a):
        plt.annotate(d, (a_roi[i], a_budget[i]))
        
    d_roi = []
    d_budget = []
    dirs = []
    for key, value in rel_dirs.items():
        dirs.append(key)
        d_roi.append(value[0])
        d_budget.append(value[1])
    
    
    plt.figure(figsize=(20, 20))
    plt.scatter(d_roi, d_budget)
    
    
    for i, d in enumerate(dirs):
        plt.annotate(d, (d_roi[i], d_budget[i]))
    