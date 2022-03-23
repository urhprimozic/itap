import pandas as pd #cvs importing
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

RANDOM_SEED = 420
df = pd.read_csv('podatki_DN1_1.csv', header=0)
columns = df.columns[1:] #1. stolpec je le dekoracijski
df = df[columns]
X = df[['x1', 'x2', 'x3', 'x4', 'x5']]
y = df['y']

from sklearn.utils import resample
k_range = list(range(1,100))
def boot(X, y,model, n_iterations):
    '''
    Z bootstrapingom oceni napako modela.
    Parameters
    ------------
    - X, y - podatki
    - model - razred z metodama fit() and predict()
    - n_iterations - št_vzorcev (=št_iteracij)

    Vrne
    -------
    Povprečno napako na vseh vzorcih (to je manj optimistična opcija)
    '''
    vzorci = [resample(X, y) for i in range(n_iterations)]
    mse = []
    for i in range(n_iterations):
        # naučiš se na enem vzorcu
        X_train = vzorci[i][0]
        y_train = vzorci[i][1]
        model.fit(X_train, y_train)
        # potestiraš na ostalih
        # i-ta napaka je mse na ostalih vzorcih
        # ker so vsi vzorci enako veliki (veliki so |X|)
        # dobimo enako, če seštejemo vse napake na vzorcih in delimo s št_ ostalih vzorccev
        err = 0
        for j in range(n_iterations):
            if j == i:
                continue
            X_test = vzorci[j][0]
            y_test = vzorci[j][1]
            err += mean_squared_error(y_test, model.predict(X_test))
        err /= n_iterations -1
        mse.append(err)
    return  np.mean(mse)

print(boot(X, y, KNeighborsRegressor(n_neighbors=1), 100))