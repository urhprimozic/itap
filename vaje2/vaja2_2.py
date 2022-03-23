# Ustvari umetno množico podatkov, na kateri boš
# testiral/a metodo logistične regresije. Ciljna
# spremenljivka naj bo nominalna z vrednostma "0" in "1".
# Razdeli podatkovje na učno in testno množico
# (train test in test set) v razmerju 4:1.
# Izdelaj posplošeni linearni model (tj. uporabi
# logistično regresijo) za napoved ciljne spremenljivke
#  (train(..., method='glm')).
# Poskušaj ciljno spremenljivko napovedati še z
# "naivnim" modelom linearne regresije, ki napoveduje
#  numerični vrednosti 0 in 1 namesto razredov "0" in "1".
#  Za pretvorbo numeričnih napovedi v nominalne uporabi
# prag 0,5. Primerjaj obe metodi na testni množici.
# BONUS: preizkusi logistično regresijo še na podatki.csv
#  iz prvih vaj ter primerjaj rezultate z metodo
# najbližjih sosedov. Je pri linearnih modelih
# skaliranje podatkov še vedno ključnega pomena?
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

RANDOM_SEED = 420

# isto kot v rešitvah
N = 5000
m = 10
X = np.random.normal(size=(N, m))
y = 1 + X[:, [0]] - 2 * X[:, [1]] + (X[:, [3]] * X[:, [4]] - X[:, [2]] * X[:, [
    5]]) / 3 + np.random.normal(0, 0.05, size=(N, 1))

# pretvorba v diskretno spremenljivko
m = np.mean(y)
z = np.copy(y)
z[y > m] = 1
z[y < m] = 0
y = z[:, 0]
#y = y.astype(np.int)
# patritioning
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.8, stratify=y, random_state=RANDOM_SEED)
# logistična regresija
reg = LogisticRegression()
reg.fit(X_train, y_train)
# točnost
predictions = reg.predict(X_test)
#keri glup si če hočeš mse na faking klasiikaciji dj misl no valda dobiš malo npako asfuck no
#mse = mean_squared_error(y_test, predictions)
napaka = np.mean(predictions == y_test)
#print('MSE = ', mse)
print('točnost : ', napaka)