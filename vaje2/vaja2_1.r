library(caret)

kakovostModela = function(model, data){
  napovedi = predict(model, newdata = data)
  mean((napovedi - data$y)^2)
}


set.seed(123)


# Ena moznost

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

# Druga moznost

# xs = matrix(rnorm(m * N), ncol = m)
# y = 1 + xs[, 1] - 2 * xs[, 2] + ...
# data = as.data.frame(xs)
# data$y = y
# stolpci = character(m + 1)
# for (i in 1:m){
#     stolpci[i] = sprintf("x%d", i)
# }
# stolpci[m + 1] = "y"
# colnames(data) = stolpci

ucniIndeksi = createDataPartition(podatki$y, p = 0.8,
                                  list=FALSE)
ucna = podatki[ucniIndeksi,]
testna = podatki[-ucniIndeksi,]

znacilke = colnames(podatki)
znacilke = znacilke[znacilke != "y"]

model0 = train(y ~ ., data=podatki, method="lm")

indeksiPoz = which(model0$finalModel$coefficients[-1] > 0)
indeksiNeg = which(model0$finalModel$coefficients[-1] < 0)

print(sprintf("Pozitivno korelirane so %s",
              paste(colnames(podatki)[indeksiPoz], collapse = ", ")
)
)
print(sprintf("Negativno korelirane so %s",
              paste(colnames(podatki)[indeksiNeg], collapse = ", ")
)
)

print("Zacenjamo z izbiro nazaj")
ctrl = rfeControl(functions = lmFuncs)
lmProfile <- rfe(ucna[znacilke], ucna[, 11],
                 sizes = c(1:10),
                 rfeControl = ctrl)

lmProfile

plot(lmProfile)
