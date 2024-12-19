library(vscDebugger)

# Fisher's test in a Completely Randomized Experiment
# Rubin (2005, II-7.3); Children's Television Workshop "Electric Company"

### Données originelles

elec <- read.csv2("http://www.evens-salies.com/electric.company.csv")
View(elec)
str(elec)

# Les niveaux de la variable catégorielle "city"
head(factor(elec$city))
elec$city <- factor(elec$city, levels=c("F","Y"), 
                    labels=c("Fresno","Youngstown"))
str(elec)
table(elec$city)

# Combien de paires de classes à Youngstown en grade 2 ?
table(elec$city, elec$grade)

### Application sur un sous-ensemble de 3 classes tests et contrôles

elec <- read.csv2("http://www.evens-salies.com/electric.company.short.csv")
View(elec)

# Comment faire l'imputation pour le vecteur d'affectations de départ 000111 ?
# Si la classe est D=0 (D=1) alors Y1 (Y0) pas observé, donner la valeur y0
#  (y1) à Y1 (Y0), sinon, c'est que D=1, et dans ce cas laisser y1 à Y1
elec$y1 <- ifelse(elec$treatment==0, elec$y0, elec$y1) 
elec$y0 <- ifelse(elec$treatment==1, elec$y1, elec$y0) 
View(elec)

# Les RO sous H0 et l'affectation 000111
#	RO ne change pas ici, mais pour une H0 générale (y0+\tau, y1-\tau), oui
elec$y <- elec$treatment*elec$y1+(1-elec$treatment)*elec$y0

# Différence de moyennes
mean1 <- sum(elec$y[elec$treatment==1])/3
mean0 <- sum(elec$y[elec$treatment==0])/3
meandiff <- mean1-mean0
meandiff

### Une routine automatique pour les C(6,3) affectations

choose(n=6,k=3)

# Charge la librairie des combinaisons (une seule fois)
# install.packages("combinat")
library(combinat)

# Liste les individus à traiter dans chacune des 20 combinaisons
tvector <- combn(elec$unit,3,simplify=T)
tvector

# Déclare un vecteur de 20 colonnes et le remplit de 0 (20 différences de moy.)
meandiffvector <- array(0, dim=c(1,20))
meandiffvector

# Applique H0 sur l'échantillon de départ (\tau=0, puis \tau=5, puis \tau=-5)
elec$y1 <- ifelse(elec$treatment==0, elec$y0, elec$y1) 
elec$y0 <- ifelse(elec$treatment==1, elec$y1, elec$y0) 

# Calcul des 20 statistiques
for(J in 1:ncol(tvector)) {

 # Vide le vecteur de traitement (le remplit de 0)
 elec$treatment <- 0

 # Sélection random (valeurs de t vector) des traités, les autres éléments = 0
 for(I in 1:nrow(tvector)) {
  elec$treatment[tvector[I,J]] <- 1
  }

 elec$y <- elec$treatment*elec$y1+(1-elec$treatment)*elec$y0
 mean1 <- sum(elec$y[elec$treatment==1])/3
 mean0 <- sum(elec$y[elec$treatment==0])/3
 meandiff <- mean1-mean0
 meandiffvector[1,J] <- meandiff
}
meandiffvector
hist(meandiffvector)
abline(v=5.067, col = "red", untf = FALSE)

# p-value (le premier vecteur [1,20] est celui qui donne 5.067)
frequency <- table(meandiffvector >= meandiffvector[1,20])
frequency
frequency[2]/sum(frequency)

# matched paired pour le CE1
elec <- read.csv2("http://www.evens-salies.com/electric.company.csv")
View(elec)

# Le nombre de ddl est 95 si on prend toutes les paires, 33 si que les grade 2
# 	Student est très proche de la N(0,1)
qt(0.025, 33, lower.tail=F)

# Test de Student pour matched pairs ; 
# 	Sur le groupe de contrôle
TTEST <- t.test(elec$control.pretest[elec$grade==2],
                elec$control.posttest[elec$grade==2], paired=T)
TTEST
TTEST$stderr

# Plot avec la normale
PRE <- elec$control.pretest[elec$grade==2]
POS <- elec$control.posttest[elec$grade==2]
DIF <- POS-PRE
MEAN <- mean(DIF)
MEAN/sqrt(var(DIF)/34)

STDE <- sqrt(var(DIF)) # Pour le graphique, ce n'est pas l'erreur standard !!!
STDE
hist(DIF, breaks=8, density=5, freq=FALSE, xlab="Différence de moyennes")
curve(dnorm(x, mean=MEAN, sd=STDE), col="pink", lwd=3, add=TRUE)




