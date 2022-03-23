library(caret)
data(iris)

set.seed(12)
kji = c(1:30)
trainCtrl = trainControl(method='boot', savePredictions=TRUE, number=10)
tuneGrd = data.frame(k=kji)

model = train(Species~., data=iris, method='knn', trControl=trainCtrl, tuneGrid=tuneGrd)
model

bs_osnovno = function(model){
  # ERR_BS = povrecje po vzorcenjih(napaka na primerih, ki niso bili izbrani)
  # To moramo izracunati za vsak k ...
  # Izracunamo najprej povprecje pri danem k in vzorcenju:
  tmp = aggregate(model$pred$pred==model$pred$obs,
                  list(model$pred$k, model$pred$Resample),
                  mean)
  # ERR_BS za dani k (ki je postal Group.1) dobimo s povprecenjem novonastalega stolpca x:
  ERR_BS = aggregate(tmp$x, list(tmp$Group.1), mean)    
  # Èe želimo še standardno deviacijo:
  # ERR_BS = aggregate(tmp$x, list(tmp$Group.1), function(x){c(mean(x), sd(x))})
  # ERR_BS = data.frame(as.matrix(ERR_BS))
  return(ERR_BS[, 2])
}

bs1 = function(model){
  # ERR_BS1 =
  # 1) za vsak primer izraèunaj povpreèno napako tistih modelov,
  #    ki niso uporabili tega primera pri uèenju
  # 2) izraèunaj povpreèje vrednosti, ki jih dobiš pri 1)
  
  # Za vsak primer in k:
  tmp = aggregate(model$pred$pred==model$pred$obs,
                  list(model$pred$k, model$pred$rowIndex),
                  mean)
  # Za vsak k:
  ERR_BS1 = aggregate(tmp$x, list(tmp$Group.1), mean)
  return(ERR_BS1[, 2])
}


bs632 = function(podatki, kji, bs1){
  # P(primer ni izbran pri zankanju) ---> 1 / e (= 0.368)
  # P(primer izbran) ---> 1 - 1 / e (= 0.632)
  n_m = dim(podatki)
  n = n_m[1]
  m = n_m[2]
  trainc = trainControl(method='cv',
                        index = list(fold1=1:n),
                        indexOut = list(fold1=1:n))  # trik, kako do vseh rezultatov: 1 fold  ...
  # formula: predpostavimo, da je ciljna spremenljivka zadnja
  f = as.formula(sprintf("%s ~ .", colnames(podatki)[m]))
  knnModelTrain = train(f, data=podatki, method='knn',
                        tuneGrid=data.frame(k=kji), trControl=trainc)
  ERR_train = knnModelTrain$results[,"Accuracy"]
  ERR_BS632 = exp(-1) * ERR_train + (1-exp(-1)) * bs1
  return(ERR_BS632)
}

eOsnovno = bs_osnovno(model)
e1 = bs1(model)
e632 = bs632(iris, kji, e1)

mini = min(eOsnovno, e1, e632) * 0.95
maxi = max(eOsnovno, e1, e632) * 1.05

plot(kji, eOsnovno, col="red", type="o", ylim=c(mini, maxi), xlab = "k", ylab = "toènost")
lines(kji, e1, col="blue", type="o")
lines(kji, e632, col="green", type="o")

legend(x = "topright",
       legend = c("eOsnovno", "e1", "e632"),
       col = c("red", "blue", "green"), lty = 1, lwd = 1)
