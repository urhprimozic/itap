import statistics
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
def summary(data):
    '''
    Returns R-like summary on data
    '''
    N = len(data)
    m = len(data[0])
    ans = {}
    ans['max'] = [max([data[i][j] for i in range(N)]) for j in range(m)]
    ans['min'] = [min([data[i][j] for i in range(N)]) for j in range(m)]
    ans['median'] = [statistics.median([data[i][j] for i in range(N)]) for j in range(m)]
    ans['mean'] = [statistics.mean([data[i][j] for i in range(N)]) for j in range(m)]
    ans['quantiles'] = [statistics.quantiles([data[i][j] for i in range(N)]) for j in range(m)]
    return ans


#seed
RANDOM_SEED =420

#loading data
iris = datasets.load_iris()
#iris je dict
#>>> iris.keys()
#dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename'])
print(iris.DESCR)
#n_exampels
N = len(iris.data)
#summary
print(summary(iris.data))
#colnames:
print(iris.feature_names)
print(iris.target_names)

#-----------RAZDELITEV PODATKOV-------
# sklearn.model_selection.train_test_split()
# + stratify , da ti fnt ohramn porazdelitev targeta
# data je 150 examplov s po 4 featurji
# target je 150 examplov, vsak je 0 ali 1 ali 2
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=4/5,  stratify=y, random_state=RANDOM_SEED)
#ucna = [x_train[i]+ [y_train[i]] for i in range(N)]
#testna = [x_test[i]+ [y_test[i]] for i in range(N)]

#--------------NEAREST NEIGHBOUR--------------------
# k = 5
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)
tocnost = sum(predictions == y_test)/len(y_test)
print(f'Točnost 5-nn modela je {tocnost}')

#kitajska:
predictions = classifier.predict(X_train)
tocnost = sum(predictions == y_train)/len(y_train)
print(f'Točnost na train  5-nn modela je {tocnost} (bit more 1')


#------ ISKANJE OPTIMALNEGA MODELA -----
k_range = list(range(1,100))
#weight_options = ["uniform", "distance"]
param_grid = dict(n_neighbors = k_range)
knn = KNeighborsClassifier()
grid = GridSearchCV(knn, param_grid, cv = 10, scoring = 'accuracy')
grid.fit(X_train,y_train)

print ('Najboljša tonost knn na train: ', grid.best_score_)
print ('Optimalni parametri: ',grid.best_params_)
print (grid.best_estimator_)

