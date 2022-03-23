from turtle import pos
import pandas as pd #cvs importing
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('podatki_DN1_2.csv', header=0)
columns = df.columns[1:] #1. stolpec je le dekoracijski
df = df[columns]
X = df[['x1', 'x2', 'x3', 'x4']]
y = df['y']
# pretvorim y v {0, 1}, ker lažje delam s tem
y = y.map(lambda x : 1 if x== 'da' else 0) 

# vrednosti v X spravim na (0,1)
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

#razdelim podatke
RANDOM_SEED = 420 # da mi med testiranjem doma vedno da enake rezultate
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=3/4, stratify=y, random_state= RANDOM_SEED)

def ROC_curve(X, y, model):
    '''
    Izračuna možne  tresholde
    Parametri
    ----------
    - X, y - podatki
    - model - model z metodo predict_proba()
    '''
    N = np.shape(X)[0]
    # možne thete
    thetas = model.predict_proba(X)
    tocke = []
    for theta in thetas:
        P = np.sum(y)[0]
        predictions = thetas.map(lambda x : 1 if x >= theta else 0)
        positive_y = predictions[y == 1]
        TP = np.sum(positive_y == 1)[0]
        FP = np.sum(predictions)[0] - TP
        
        TRP = TP/P 
        FPR = FP/N   