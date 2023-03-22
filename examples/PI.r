library('propagate')
library('MASS')

df <- read.csv("puro.csv")
f <- nls(y ~ (t0 * x / (t1 + x^2)), data=df, start = list(t0 = 205.1, t1 = 0.08))
#f <- nls(y ~ (t0 * x), data=df, start = list(t0 = 205.1))
summary(f)

cis <- predictNLS(f, newdata = data.frame(x = c(0.04, 0.15, 0.6)), interval='confidence', nsim=10000, alpha=0.1)
cis
pis <- predictNLS(f, newdata = data.frame(x = c(0.04, 0.15, 0.6)), interval='prediction', nsim=10000, alpha=0.1)
pis
