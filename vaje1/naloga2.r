library(caret)
podatki = read.csv("neumazani.csv", fileEncoding="UTF-8")
#str(podatki)
#View(podatki)
#summary(podatki)


# zadnje pucanje
podatki$X6 = as.numeric(as.character(podatki$X6))
# ker delamo jklasifikacijo, na Y damo 2 level factor
podatki$Y = as.factor(podatki$Y)

#View(podatki)
# razdelitefv podatkov 
indeks = createDataPartition(podatki$Y, p = 1/4, list=FALSE)
ucna = podatki[indeks,]
testna = podatki[-indeks, ]

# training
natrenirano = train(Y ~ . - X , data=ucna, method="knn")
#potestiraš na TESTNI
napovedi = predict(natrenirano, newdata=testna)
n_pravilnih  =sum(napovedi == testna$Y)
print("Delež uspeha na testni: ")
print(n_pravilnih / length(napovedi))

# to je slaba uspešnost. Razlog je,
#ker naša metoda knn ubistu upošteva le X&, 
#ki je reda 1000 večji od ustalih

#FIX: preprocess podatkov
preProcVrednosti = preProcess(ucna, method = c("center", "scale"))
ucna = predict(preProcVrednosti, ucna)
testna = predict(preProcVrednosti, testna)

natrenirano = train(Y ~ . - X , data=ucna, method="knn")
#potestiraš na TESTNI
napovedi = predict(natrenirano, newdata=testna)
n_pravilnih  =sum(napovedi == testna$Y)
print("Delež uspeha na testni z uporabo preporcessa: ")
print(n_pravilnih / length(napovedi))
