from random import seed
import pandas as pd
import numpy as np
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Ustvari umetno množico podatkov, na kateri boš
# testiral/a metodo linearne regresije. Množica naj ima
# vsaj 10 spremenljivk.
# Razdeli podatkovje na učno in testno množico v razmerju
#  4:1.
# Izdelaj linearni model za napoved ciljne
# spremenljivke (train(..., method="lm")). Kako
# posamezne spremenljivke vplivajo na model, tj.
# katere spremenljivke so pozitivno korelirane s
# ciljno spremenljivko? Katere spremenljivke so
# negativno korelirane s ciljno spremenljivko?
# Z izbiro nazaj izberi najboljših 5 spremenljivk.
# Katere spremenljivke je model izbral? Se to ujema z
#  najbolj vplivnimi spremenljivkami modela iz prejšnje točke?
# Namig: ročnemu programiranju se lahko izognemo s
# funkcijo rfe iz paketa caret, pri čemer ji v rfeControl z izbiro lmFuncs povemo, da želimo linearne modele.
# Nariši, kako se natančnost modela spreminja s številom
#  spremenljivk. Kolikšno je optimalno število
#  spremenljivk?

RADNOM_SEED = 420
np.random.seed(RADNOM_SEED)

# y definiram isto kot na vajah.
N = 5000
m = 10
X = np.random.normal(size=(N, m))
y = 1 + X[:, [0]] - 2 * X[:, [1]] + (X[:, [3]] * X[:, [4]] - X[:, [2]] * X[:, [
                                     5]]) / 3 + np.random.normal(0, 0.01, size=(N, 1))
# delitev podatkov
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=RADNOM_SEED)

reg = LinearRegression()
reg.fit(X, y)
print('Koeficienti', reg.coef_)
print('Zamik: ', reg.intercept_)
i_positive = np.where(reg.coef_[0] > 0)
i_not_related = np.where(reg.coef_[0] == 0)
i_negative = np.where(reg.coef_[0] < 0)

# probamo najbolj glup feature ranking:
# izbira nazaj - RFE (potrebuje #ikzbranih featurjev)
# in RFECV - sam najde optimaalno število
#estimator = SVR(kernel="linear")
estimator = LinearRegression()
selector = RFECV(estimator, step=1)
selector.fit(X_train, y_train)
print('ranking: ', selector.ranking_)
print('selected features: ', selector.get_support(indices=True))
# res dela dost similar kot resitev v R :)