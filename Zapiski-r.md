# Zapiski R 
### Osnovni podatki

    summary(data) # vse
    View(data)
Dostop do stolpca 

    data$imse_stolpca
### Delitev podatkov
    i = createPartition(stolpec_ki_se_ga_upošteva,  p = delež_učnih, list=FALSE)
    ucna = data[i,]
    testna  =data[-i,]

### Training
    natrenirano = train(stolpci_y ~ stolpci_x, data= ucna, method=metoda, tuneGrid=če si bl pametn ko R)

### Prediction
    predict(natrenirano, newdata = testna)  

### Error analisyssdfčjhnes

