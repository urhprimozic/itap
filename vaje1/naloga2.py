# Naloži podatke v datoteki podatki.csv.
# Če se bo R pritožil, da datoteka ne obstaja,
# pokliči getwd().
# Podatke analiziraj tako kot v prvem delu.
# Pričakuj, da postopek ne bo povsem enostaven,
# saj se v podatkih skriva nekaj napak.
# Namig: pomagamo si lahko s funkcijama View in str.
# Kako dobre so napovedi modela? Lahko natančnost
#  kako izboljšamo?
# Namig: še enkrat poglejmo porazdelitve posameznih
#  spremenjivk ter razmislimo, kako deluje metoda
# najbližjih sosedov.

import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from util import summary

RANDOM_SEED = 420

df = pd.read_csv('neumazani.csv', header=0)
# 1. stolpec je le dekoracija
columns = df.columns[1:]
df = df[columns]
# summary
print(df.columns.values)
print(summary(df.to_numpy()))

# X4 nima numeričnih vrednosti
df['X4'] = df['X4'].map(lambda x: 1 if x == 'DA' else 0)

# deitev podatkov
X = df[['X1', 'X2', 'X3', 'X4', 'X5', 'X6']]
y = df['Y']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=4/5,  stratify=y, random_state=RANDOM_SEED)

# knn:
k_range = list(range(1, 100))
knn = KNeighborsClassifier()
param_grid = dict(n_neighbors=k_range)
# grid with 10 folkd cross validation
grid = GridSearchCV(knn, param_grid,  cv=10, scoring='accuracy')
grid.fit(X_train, y_train)

print('Najboljša tonost knn na train: ', grid.best_score_)
print('Optimalni parametri: ', grid.best_params_)
# print(grid.best_estimator_)

# dobiš 0.64., kar je slaba točnost, ker
# podatki niso normalizirani
# uporabimo preprocess, tokrat minmax scaler in vse dam na (0,1)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

# novi podatki
grid.fit(X_train, y_train)
print('Najboljša tonost knn na train: ', grid.best_score_)
print('Optimalni parametri: ', grid.best_params_)
#dobiš 0.875, z R so dobil 0.89