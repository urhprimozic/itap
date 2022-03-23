
# Naloži podatke podatki_vaja3.csv in uporabi linearno 
# regresijo za učenje modela.
# Podatkom dodaj še stolpce drugega reda, torej produkte
#  vseh parov začetnih stolpce.
# Konstruiraj linearna modela, ki sprejmeta osnovne 
# podatke in podatke, razširjene do reda 2. Napako obeh 
# modelov oceni z metodo 10-kratnega prečnega 
# preverjanja. Kateri deluje bolje?
# Napiši funkcijo razsiriPodatke(podatki, red), ki nabor 
# spremenjivk razširi do poljubnega reda.
# Namig: pomagamo si lahko s funkcijo poly.
# Konstruiraj še linearne modele za še dodatno 
# razširjene podatke. Nariši graf, odvisnosti napake
#  (npr. RMSE) od stopnje razširitve.
# Grafu dodaj še izračun napake na učni množici.
#  Kaj se dogaja z optimizmom ocene napake?
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.preprocessing import PolynomialFeatures

df = pd.read_csv('podatki_vaja3.csv', header=0)
# 1. stolpec je le dekoracija
columns = df.columns[1:]
df = df[columns]
X1 = df[['x1', 'x2', 'x3']]
y = df['y']


# v sklearn razširiš zadeve z mogočno funcijo 
#PolynomialFeatures
poly = PolynomialFeatures(2)
#PAZI! DODA TUD FEATURE 1
X2 = poly.fit_transform(X1)[:, 1:]
reg = LinearRegression()
results1 = cross_validate(reg, X1, y, cv=3, scoring=('neg_mean_absolute_error'))
results2 = cross_validate(reg, X2, y, cv=3, scoring=('neg_mean_absolute_error'))
print(-np.mean(results1['test_score']))
print(-np.mean(results2['test_score']))
#dobimo
#0.13844413183191404
#0.08874524157738022

#ful bolš je če dodamo featurje (no shit)