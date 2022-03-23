library(caret)


razsiriPodatke = function(podatki, stopnja){
  if (stopnja == 1){
    return(podatki)
  }
  n_vhodne = 
    razsirjeni = data.frame(poly(as.matrix(podatki[, 1 : length(podatki)-1]),
                                 degree=stopnja,
                                 raw=T),  # da izklopimo ortogonalnost
                            y=podatki$y)
}

#setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
podatki = read.csv("podatki_vaja3.csv")

podatki2 = razsiriPodatke(podatki, 2)


ctrl = trainControl(method = "cv", number = 10, savePredictions = T)
lin = train(y ~ ., data = podatki, method = "lm", trControl = ctrl)
lin2 = train(y ~ ., data = podatki2, method = "lm", trControl = ctrl)
print(lin)
print(lin2)


maxRed = 10
# pozor: stevilo spremenljivk hitro raste ...
# DN: koliko jih je pri redu r, ce jih je na zacetku n?
napake = rep(0.0, maxRed)
napakeUcna = rep(0.0, maxRed)

for (red in 1:maxRed){
  podatkiRed = razsiriPodatke(podatki, red)
  lin = train(y ~ ., data = podatkiRed, trControl = ctrl, method = "lm")
  napake[red] = lin$results$RMSE
  napakeUcna[red] = RMSE(predict(lin, newdata = podatkiRed), podatki$y)
}

mini = min(napake, napakeUcna) * 0.95
maxi = max(napake, napakeUcna) * 1.05

plot(1:maxRed, napake,
     xlab = "red", ylab = "RMSE",
     type="o", col="blue",
     ylim=c(mini, maxi))
lines(1:maxRed, napakeUcna,
      xlab = "red", ylab = "RMSE",
      type="o", col="red")
legend(x = "topright",
       legend = c("pre�no preverjanje", "u�na mno�ica"),
       col = c("blue", "red"), lty = 1, lwd = 1)

