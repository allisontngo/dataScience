setwd("data/")
carseats = load("Carseats.rda")
fix(Carseats)
write.csv(Carseats, file="carseats.csv")