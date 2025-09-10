# package used to import spss file
library(foreign)

don.desbois<-read.spss("desbois.sav",
to.data.frame = TRUE)

# change format of some variables
edit(don.desbois)

# cr?ation de la variable ? expliquer de type numeric
Y<-ifelse(don.desbois$DIFF=="1",0,1)
N<-length(Y)

x<-data.frame(don.desbois[,-c(1,2,8)])

x$STATUS<- as.factor(x$STATUS)
x$ToF<- as.factor(x$ToF)
x$OWNLAND<- as.factor(x$OWNLAND)
##############################################################
######  Exploration ##########################################
##############################################################

tab.bi <- table(don.desbois$STATUS,don.desbois$DIFF)

row.profil<-round(sweep(addmargins(tab.bi,2,
list(All=sum)),1,apply(tab.bi,1,sum)/100,"/"),1)

plot(tab.bi)

barplot(table(don.desbois$ToF,don.desbois$DIFF))

boxplot(x$r1~Y) # variable ? tester r1, r3, r5, r6, r7, r11,
# r12, r14, r17, r19, r28, r36, r37

##############################################################
######    CHOIX d'un mod?le logistique  #########################
##############################################################

library(MASS)
library(BMA)

# choix avec la fonction bic.glm
choix.bic.glm <- bic.glm (x,Y,strict = FALSE, OR = 20,
data=x, glm.family="binomial",factor.type=TRUE)

summary (choix.bic.glm,conditional=T,digits=2)
imageplot.bma(choix.bic.glm)


# Mod?le ? 6 variables : STATUS + HECTARE + r1 + r3 + r17 + r24 + r28 + r36

# choix avec la fonction step en prenant le crit?re BIC
choix.step.glm<-stepAIC(glm(Y ~ . ,data=x[,-c(1:5)], family="binomial"),k=log(N))

 summary(choix.step.glm)
 
 # Mod?le : STATUS + HECTARE + r1 + r3 + r17 + r19 + r22 + r36 + r37
##############################################################
######    REGRESSION LOGISTIQUE      #########################
##############################################################

# Function qui va construire un mod?le logistique sur  
# un ?chantillon d'apprentissage, puis qui calcule sur un
# ?chantillon test le taux de mal class?
# Cette op?ration est r?p?t?e un certain nombre de fois 
# afin d'obtenir un taux moyen de mal class? 
# A chaque fois, on choisit le meilleur seuil possible pour
# attribuer la valeur oui/non au score estim?


val.croisee.logistic<-function(repl)
{
# arguments de sortie  :
# le meilleur seuil :
best.seuil<-matrix(rep(0,2*repl),repl,2)
# le taux de mal class?
best.mc<- matrix(rep(0,2*repl),repl,2)

# initialisatio,
N<-length(Y)
n<-round(N*0.8)

for (k in 1:repl)
 {# construction d'un ?chantillon d'apprentissage et test
  set.seed(k)
  samp<-sample(1:N,n)
  ech.app<-x[samp,]
  ech.test<-x[-samp,]
  y.app<-Y[samp]
  y.test<-Y[-samp]

  # Le mod?le de Dominique Desbois sur l'?chantillon d'apprentissage
  res.glm<-glm(y.app~STATUS + HECTARE + r1 + r3 + r17 + r28 + r36,family=binomial(link=logit),data=ech.app)
  res.glm.desbois<-glm(y.app~r1+r12+r14+r17+r32+r36,family=binomial(link=logit),data=ech.app)

  # Les pr?dictions du mod?le logistique sur l'?chantillon test
  eta<-predict(res.glm,newdata=ech.test)
  mu<-exp(eta)/(1+exp(eta))
  eta.desbois<-predict(res.glm.desbois,newdata=ech.test)
  mu.desbois<-exp(eta.desbois)/(1+exp(eta.desbois))
  
  # Le choix du seuil qui minimise le taux d'erreur : (Sum des bien class?)/n
  seuil<-seq(0.01,0.99,0.01)
  mc<-rep(0,99)
  mc.desbois<-rep(0,99)
  for (i in 1:99)
  {
   b<-table(mu>seuil[i],y.test)
   b.desbois<-table(mu.desbois>seuil[i],y.test)
   mc[i]<-(b[1]+b[4])/(N-n)
   mc.desbois[i]<-(b.desbois[1]+b.desbois[4])/(N-n)
  }

  best.seuil[k,1]<-mean(seuil[which(mc==max(mc,na.rm=TRUE))[1]])
  best.seuil[k,2]<-mean(seuil[which(mc.desbois==max(mc.desbois,na.rm=TRUE))[1]])
  best.mc[k,1]<-mc[which(mc==max(mc,na.rm=TRUE))[1]]
  best.mc[k,2]<-mc.desbois[which(mc.desbois==max(mc.desbois,na.rm=TRUE))[1]]
 }
 return(list(best.mc=best.mc,best.seuil=best.seuil))
} 

res.logistic<-val.croisee.logistic(500)
hist(res.logistic$best.mc,probability=T,col='lightblue',border="pink",main="",xlab='Distribution des taux de mal class?s',cex=0.8,cex.lab=0.66)

plot(density(res.logistic$best.mc[,1]),col="red",lwd=2)
lines(density(res.logistic$best.mc[,2]),col="blue",lty=2,lwd=2)
legend("topleft",c("model Tibo","Model Dom"),col=c("red","blue"),lty=c(1,2))
##############################################################
######     bagging         #########################
##############################################################

#library(ipred)
library(adabag)

val.croisee.bagging<-function(repl,mfinal)
{
# arguments de sortie  :

# le taux de mal class?
best.mc<-rep(0,repl)

# initialisation
N<-length(Y)
n<-round(N*0.8)

for (k in 1:repl)
 {# construction d'un ?chantillon d'apprentissage et test
  set.seed(k)
  samp<-sample(1:N,n)
  ech.app<-data.frame(x[samp,],as.factor(Y[samp]))
  names(ech.app)[28]<-"Y"
  ech.test<-data.frame(x[-samp,],as.factor(Y[-samp]))
  names(ech.test)[28]<-"Y"
 
  # Fonction bagging 
  res.bag<-bagging(Y~.,data=ech.app,mfinal=mfinal)
  mu<-predict(res.bag,newdata=ech.test)

  b<-table(mu$class,ech.test$Y)
  best.mc[k]<-(b[1]+b[4])/(N-n)
  
 }
 return(list(best.mc=best.mc))
} 

res.bagging<-val.croisee.bagging(500,100)

plot(density(res.bagging$best.mc[,1]),col="red",lwd=2)
lines(density(res.bagging$best.mc[,2]),col="blue",lty=2,lwd=2)
legend("topleft",c("logistic","bagging"),col=c("red","blue"),lty=c(1,2))

hist(res.bagging$best.mc,probability=T,col='lightblue',border="pink",main="",xlab='Distribution des taux de mal class?s',cex=0.8,cex.lab=0.66)
lines(density(res.bagging$best.mc),col="red",lwd=2)

# Comparaison des m?thodes
apply(cbind(res.logistic[[1]],res.bagging[[1]]),2,mean)
apply(cbind(res.logistic[[1]],res.bagging[[1]]),2,sd)

##############################################################
######    FORET ALEATOIRE          #########################
##############################################################

# Random Forest
# Param?tres pris par d?faut ....

library(randomForest)
library(e1071)

val.croisee.randomforest<-function(repl)
{
# arguments de sortie  :
# le meilleur seuil :
best.seuil<-rep(0,repl)
# le taux de mal class?
best.mc<- rep(0,repl)

# Avec les seules variables quantitatives :
N<-length(Y)
n<-round(N*0.8)

for (k in 1:repl)
 {# construction d'un ?chantillon d'apprentissage et test
 # construction d'un ?chantillon d'apprentissage et test
  set.seed(k)
  samp<-sample(1:N,n)
  ech.app<-don.desbois[samp,]
  ech.test<-don.desbois[-samp,]
  y.app<-Y[samp]
  y.test<-Y[-samp]
  
  fit=randomForest(y.app~r1+r12+r14+r17+r32+r36,data=ech.app,
  xtest=ech.test[,c("r1","r12","r14","r17","r32","r36")],
  ytest=y.test,do.trace=20,
  importance=TRUE,norm.vote=FALSE)

  mu<-fit$test$predicted

  # Le choix du seuil qui minimise le taux d'erreur : (Sum des bien class?)/n
  seuil<-seq(0.01,0.99,0.01)
  mc<-rep(0,99)

  for (i in 1:99)
  {
   b<-table(mu>seuil[i],y.test)
   mc[i]<-(b[1]+b[4])/(N-n)
  }
  best.seuil[k]<-mean(seuil[which(mc==max(mc,na.rm=TRUE))])
  best.mc[k]<-mc[which(mc==max(mc,na.rm=TRUE))[1]]
 }
 return(list(best.mc=best.mc,best.seuil=best.seuil))
}

res.randomforest<-val.croisee.randomforest(100)
hist(res.randomforest$best.mc,probability=T,col='lightblue',border="pink",main="",xlab='Distribution des taux de mal class?s',cex=0.8,cex.lab=0.66)
lines(density(res.randomforest$best.mc),col="red",lwd=2)

# Comparaison des m?thodes
apply(cbind(res.logistic[[1]],res.bagging[[1]],res.randomforest[[1]]),2,mean)
apply(cbind(res.logistic[[1]],res.bagging[[1]],res.randomforest[[1]]),2,sd)

##############################################################
######    Support Vecteur Machine        #########################
##############################################################

val.croisee.svm<-function(repl)
{
# arguments de sortie  :
# le meilleur seuil :
best.seuil<-rep(0,repl)
# le taux de mal class?
best.mc<- rep(0,repl)

# Avec les seules variables quantitatives :
N<-length(Y)
n<-round(N*0.8)

for (k in 1:repl)
 {# construction d'un ?chantillon d'apprentissage et test
 # construction d'un ?chantillon d'apprentissage et test
  set.seed(k)
  samp<-sample(1:N,n)
  ech.app<-don.desbois[samp,]
  ech.test<-don.desbois[-samp,]
  y.app<-Y[samp]
  y.test<-Y[-samp]

  fit=svm(y.app~r1+r12+r14+r17+r32+r36,data=ech.app)
  mu<-predict(fit,newdata= ech.test)

  # Le choix du seuil qui minimise le taux d'erreur : (Sum des bien class?)/n
  seuil<-seq(0.01,0.99,0.01)
  mc<-rep(0,99)

  for (i in 1:99)
  {
   b<-table(mu>seuil[i],y.test)
   mc[i]<-(b[1]+b[4])/(N-n)
  }
  best.seuil[k]<-mean(seuil[which(mc==max(mc,na.rm=TRUE))])
  best.mc[k]<-mc[which(mc==max(mc,na.rm=TRUE))[1]]
 }
 return(list(best.mc=best.mc,best.seuil=best.seuil))
}

res.svm<-val.croisee.svm(100)
hist(res.svm$best.mc,probability=T,col='lightblue',border="pink",main="",xlab='Distribution des taux de mal class?s',cex=0.8,cex.lab=0.66)
lines(density(res.svm$best.mc),col="red",lwd=2)

# Comparaison des m?thodes
apply(cbind(res.logistic[[1]],res.bagging[[1]],res.randomforest[[1]],res.svm[[1]]),2,mean)
apply(cbind(res.logistic[[1]],res.bagging[[1]],res.randomforest[[1]],res.svm[[1]]),2,sd)


##############################################################
######    Generalized Additive Model    ######################
##############################################################

library(gam)

val.croisee.gam<-function(repl)
{
# arguments de sortie  :
# le meilleur seuil :
best.seuil<-rep(0,repl)
# le taux de mal class?
best.mc<- rep(0,repl)

# Avec les seules variables quantitatives :
N<-length(Y)
n<-round(N*0.8)

for (k in 1:repl)
 {# construction d'un ?chantillon d'apprentissage et test
 # construction d'un ?chantillon d'apprentissage et test
  set.seed(k)
  samp<-sample(1:N,n)
  ech.app<-don.desbois[samp,]
  ech.test<-don.desbois[-samp,]
  y.app<-Y[samp]
  y.test<-Y[-samp]
  
  fit=gam(y.app~r1+s(r12)+r14+s(r17)+s(r32)+s(r36),family=binomial(link=logit),data=ech.app)

  # Les pr?dictions du mod?le gam sur l'?chantillon test
  eta<-predict(fit,newdata=ech.test)
  mu<-exp(eta)/(1+exp(eta))

  # Le choix du seuil qui minimise le taux d'erreur : (Sum des bien class?)/n
  seuil<-seq(0.01,0.99,0.01)
  mc<-rep(0,99)

  for (i in 1:99)
  {
   b<-table(mu>seuil[i],y.test)
   mc[i]<-(b[1]+b[4])/(N-n)
  }
  best.seuil[k]<-mean(seuil[which(mc==max(mc,na.rm=TRUE))])
  best.mc[k]<-mc[which(mc==max(mc,na.rm=TRUE))[1]]
 }
 return(list(best.mc=best.mc,best.seuil=best.seuil))
}

res.gam<-val.croisee.gam(100)
hist(res.gam$best.mc,probability=T,col='lightblue',border="pink",main="",xlab='Distribution des taux de mal class?s',cex=0.8,cex.lab=0.66)
lines(density(res.gam$best.mc),col="red",lwd=2)

# Comparaison des m?thodes
apply(cbind(res.logistic[[1]],res.bagging[[1]],res.randomforest[[1]],res.svm[[1]],res.gam[[1]]),2,mean)

mc.mod<-c(res.logistic[[1]],res.bagging[[1]],res.randomforest[[1]],res.svm[[1]],res.gam[[1]])
type.mod<-c(rep("1",100),rep("2",100),rep("3",100),rep("4",100),rep("5",100))

boxplot(mc.mod~type.mod,names=c("logistic","bagging","randomforest","svm","gam"))

apply(cbind(res.logistic[[1]],res.bagging[[1]],res.randomforest[[1]],res.svm[[1]],res.gam[[1]]),2,sd)


##############################################################
######    Non param?trique   ######################
##############################################################

library(np)

val.croisee.np<-function(repl)
{
# arguments de sortie  :
# le meilleur seuil :
best.seuil<-rep(0,repl)
# le taux de mal class?
best.mc<- rep(0,repl)

# Avec les seules variables quantitatives :
N<-length(Y)
n<-round(N*0.8)

for (k in 1:repl)
 {# construction d'un ?chantillon d'apprentissage et test
 # construction d'un ?chantillon d'apprentissage et test
  set.seed(k)
  samp<-sample(1:N,n)
  ech.app<-don.desbois[samp,]
  ech.test<-don.desbois[-samp,]
  y.app<-Y[samp]
  y.test<-Y[-samp]

  fit=npconmode(y.app~r1+r12+r14+r17+r32+r36,tol=.1, ftol=.1,data=ech.app)

  # Les pr?dictions du mod?le gam sur l'?chantillon test
  eta<-predict(fit,newdata=ech.test)
  mu<-exp(eta)/(1+exp(eta))

  # Le choix du seuil qui minimise le taux d'erreur : (Sum des bien class?)/n
  seuil<-seq(0.01,0.99,0.01)
  mc<-rep(0,99)

  for (i in 1:99)
  {
   b<-table(mu>seuil[i],y.test)
   mc[i]<-(b[1]+b[4])/(N-n)
  }
  best.seuil[k]<-mean(seuil[which(mc==max(mc,na.rm=TRUE))])
  best.mc[k]<-mc[which(mc==max(mc,na.rm=TRUE))[1]]
 }
 return(list(best.mc=best.mc,best.seuil=best.seuil))
}

res.gam<-val.croisee.gam(100)
hist(res.gam$best.mc,probability=T,col='lightblue',border="pink",main="",xlab='Distribution des taux de mal class?s',cex=0.8,cex.lab=0.66)
lines(density(res.gam$best.mc),col="red",lwd=2)

# Comparaison des m?thodes
apply(cbind(res.logistic[[1]],res.bagging[[1]],res.randomforest[[1]],res.svm[[1]],res.gam[[1]]),2,mean)

mc.mod<-c(res.logistic[[1]],res.bagging[[1]],res.randomforest[[1]],res.svm[[1]],res.gam[[1]])
type.mod<-c(rep("1",100),rep("2",100),rep("3",100),rep("4",100),rep("5",100))

boxplot(mc.mod~type.mod,names=c("logistic","bagging","randomforest","svm","gam"))

apply(cbind(res.logistic[[1]],res.bagging[[1]],res.randomforest[[1]],res.svm[[1]],res.gam[[1]]),2,sd)
