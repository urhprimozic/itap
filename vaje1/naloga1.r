library(caret)
data(iris)
#-----------------INFO------------------
# dimenzije
print(dim(iris))
# imena stolpcev
print(colnames(iris))
# št pripadnikov vrste
print(table(iris["Species"]))
# osnovne informacije (min, max, kvartili, mediana, mean, ...)
print(summary(iris))
 
#----------------RAZDELITEV PODATKOV--------
# za razdelitefv VEDNO uporabiš narejene funckije. 
# TUkaj hočeš zajeti vse vrste - narediš partitio na Species
indeks_ucna = createDataPartition(iris$Species, p=4/5, list=FALSE)
ucna = iris[indeks_ucna,]  # list=FALSE, da se tukaj pri indeksiranju ne sesuje
testna = iris[-indeks_ucna,]  # negativen indeksi = vzemi vse razen njih

#------------UČENJE-----------

# KAKO POISKATI OPTIMALEN K ZA KNN ?  (glede na ucno al kaj)
    # funckija train(y_1 + ... + y_n ~x_1 + ... + x_p , data=, method = ) 
    # pripravi vse potrebno za učenje y(x) iz potakov data s pomočjo methode method,
    # pofita vsak model in ozračuna resampling based performance measure 

    # natrenirano = train(Species ~ Sepal.Width + Sepal.Length + Petal.Width + Petal.Length, data=ucna, method="knn")
    # kdo bo pa pisal vsa imena stolpcev na roke ...
    natrenirano = train(Species ~ ., data=ucna, method="knn")  # . = vsi stolpci v podatkih, razen ze omenjenih v formuli
    print(natrenirano)

# ROČNA UPORABA POLJUBNEGA K-JA
    # dokumentacija: natrenirano = train(Species ~ ., data=trainData, k=1, method="knn") - NE DELA
    # Uporabit moraš tuneGrid. Lahko se sprehodno čez 120 možnih k, pa poplotamo kappa (accuracy) v odvnisnosti od k
    natrenirano_rocno = train(Species ~ ., data=ucna, method="knn", tuneGrid = data.frame(k = 2:12))
    plot(natrenirano_rocno)
    plot(natrenirano_rocno$results$k, natrenirano_rocno$results$Kappa, type="o", col="green", xlab="k", ylab="Kappa")
# Hm, kaj pa sploh je Kappa ? ;)
# Pojasni, zakaj tocnost pade ko je k vse vecji --> bližaš se modelu, ki zmeraj vrne isto

#---------------NAPOVEDI--------

# VEDNO RABIŠ TRAIN NA UCNI, IN NEWDATA NA DISJUNKTNI TESTNI!!!!

napovedi = predict(natrenirano, newdata = testna)  # in ne natrenirano$finalModel
n_pravilnih = sum(napovedi == testna$Species)
print(sprintf("Tocnost modela na testni: %.3f", n_pravilnih / length(napovedi)))

#CHINA WAY:
sistem_te_laze = train(Species ~., data=testna, method = "knn")
ne_verjuj_sta_ti_kaze = predict(sistem_te_laze, newdata=testna)
n_kao_pravilnih = sum(sistem_te_laze == testna$Species)
print(sprintf("Če goljufamo, dobimo tako točnost:: %.3f", n_kao_pravilnih / length(sistem_te_laze)))