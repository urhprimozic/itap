#Ustvari umetno množico podatkov, na kateri boš testiral/a metodo linearne regresije. Množica naj ima vsaj 10 spremenljivk.
#Razdeli podatkovje na učno in testno množico v razmerju 4:1.
#Izdelaj linearni model za napoved ciljne spremenljivke (train(..., method="lm")). Kako posamezne spremenljivke vplivajo na model, tj. katere spremenljivke so pozitivno korelirane s ciljno spremenljivko? Katere spremenljivke so negativno korelirane s ciljno spremenljivko?
#Z izbiro nazaj izberi najboljših 5 spremenljivk. Katere spremenljivke je model izbral? Se to ujema z najbolj vplivnimi spremenljivkami modela iz prejšnje točke?
#Namig: ročnemu programiranju se lahko izognemo s funkcijo rfe iz paketa caret, pri čemer ji v rfeControl z izbiro lmFuncs povemo, da želimo linearne modele.
#Nariši, kako se natančnost modela spreminja s številom spremenljivk. Kolikšno je optimalno število spremenljivk?

library(caret)
#Seed nastavim na določeno ksntanto  da zmeraj dobin enake rezultate
set.seed(123)

# y je linearna kombinacija X in nek šum
N = 5000 
m = 10
x1 = rnorm(N)
x2 = rnorm(N)
x3 = rnorm(N)
x4 = rnorm(N)
x5 = rnorm(N)
x6 = rnorm(N)
x7 = rnorm(N)
x8 = rnorm(N)
x9 = rnorm(N)
x10 = rnorm(N)
y = 1 + x1 - 2 * x2 + (x4 * x5 - x3 * x6) / 3 + rnorm(N, 0, 0.01)
podatki = data.frame(x1 = x1, x2 = x2, x3 = x3,
                     x4 = x4, x5 = x5, x6 = x6,
                     x7 = x7, x8 = x8, x9 = x9,
                     x10 = x10, y = y)

indeks = createDataPartition(podatki$y, p = 0.8, list=FALSE)
ucna = podatki[indeks,]
testna = podatki[-indeks,]

lin_model_1 = train(y ~., data = ucna, method="lm")

# linearen model - najbolj pomembna je korelacija

indeksiPoz = which(lin_model_1$finalModel$coefficients[-1] > 0)
indeksiNeg = which(lin_model_1$finalModel$coefficients[-1] < 0)