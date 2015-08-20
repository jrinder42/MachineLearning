library(ipred)
library(randomForest)
library(rpart)
library(ROCR)
library(gbm)
library(nnet)
library(e1071)
library(caret)
library(party)
library(MASS)

# Reading in the data
training <- read.csv(url("http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv"))
testing <- read.csv(url("http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/test.csv"))

# Join together the test and train sets for easier feature engineering
testing$Survived <- NA
combo <- rbind(training, testing)

# Convert to a string
combo$Name <- as.character(combo$Name)

# Engineered variable: Title
combo$Title <- sapply(combo$Name, FUN = function(x) {strsplit(x, split='[,.]')[[1]][2]})
combo$Title <- sub(' ', '', combo$Title)

# Combine small title groups
combo$Title[combo$Title %in% c('Mme', 'Mlle')] <- 'Mlle'
combo$Title[combo$Title %in% c('Capt', 'Don', 'Major', 'Sir')] <- 'Sir'
combo$Title[combo$Title %in% c('Dona', 'Lady', 'the Countess', 'Jonkheer')] <- 'Lady'

# Convert to a factor
combo$Title <- factor(combo$Title)

# Engineered variable: Family size
combo$FamilySize <- combo$SibSp + combo$Parch + 1


# Engineered variable: Family
combo$Surname <- sapply(combo$Name, FUN = function(x) {strsplit(x, split='[,.]')[[1]][1]})
comboSurname <- combo$Surname
combo$FamilyID <- paste(as.character(combo$FamilySize), combo$Surname, sep="")
combo$FamilyID[combo$FamilySize <= 2] <- 'Small'
# Delete erroneous family IDs
famIDs <- data.frame(table(combo$FamilyID))
famIDs <- famIDs[famIDs$Freq <= 2,]
combo$FamilyID[combo$FamilyID %in% famIDs$Var1] <- 'Small'
# Convert to a factor
# New
combo$FamilyID[combo$FamilySize <= 5 & combo$FamilySize >= 3] <- 'Medium'
combo$FamilyID[combo$FamilySize >= 6] <- 'Large'

combo$FamilyID <- factor(combo$FamilyID)



# Fill in Age NAs
summary(combo$Age)
Agefit <- rpart(Age ~ Pclass + Sex + SibSp + Parch + Fare + Embarked + Title + FamilySize, 
                data = combo[!is.na(combo$Age), ], method="anova")
combo$Age[is.na(combo$Age)] <- predict(Agefit, combo[is.na(combo$Age),])
# Check what else might be missing
summary(combo)
# Fill in Embarked blanks
summary(combo$Embarked)
which(combo$Embarked == '')
combo$Embarked[c(62,830)] = "S"
combo$Embarked <- factor(combo$Embarked)
# Fill in Fare NAs
summary(combo$Fare)
which(is.na(combo$Fare))
combo$Fare[1044] <- median(combo$Fare, na.rm = T)



# Split back into test and train sets
trainOrigin <- combo[1:1891, ] # Hold original training set
testOrigin <- combo[892:1309, ] # Hold original testing set
training <- combo[1:891, -c(1, 4, 9, 11, 15)]
# Column two is the Survived column
testing <- combo[892:1309, -c(1, 2, 4, 9, 11, 15)]

inTraining <- createDataPartition(y = training$Survived, p = 0.75, list = F)
subTraining <- training[inTraining, ]
validation <- training[-inTraining, ]

set.seed(4872)

# CART (Classification and Regression Tree) - library(rpart) 
fit.rpart <- rpart(Survived ~., data = subTraining, control = rpart.control(minsplit=50, cp=0))
pred.rpart <- predict(fit.rpart, validation)

predCART <- prediction(pred.rpart, validation$Survived)
roc.perfCART = performance(predCART, measure = "tpr", x.measure = "fpr")
plot(roc.perfCART, col = "red", lwd = 2)
abline(a=0,b=1,lwd=2,lty=2,col="gray")

# Function for finding the optimal cutoff 
acc.perfCART = performance(predCART, measure = "acc")
plot(acc.perfCART, col = "blue")
indCART = which.max( slot(acc.perfCART, "y.values")[[1]] )
accCART = slot(acc.perfCART, "y.values")[[1]][indCART]
cutoffCART = slot(acc.perfCART, "x.values")[[1]][indCART]
print(c(accuracy= accCART, cutoff = cutoffCART))
#auc.perfCART = performance(predCART, measure = "auc")
#print(auc.perfCART)

pred.rpart <- ifelse(pred.rpart >= cutoffCART, 1, 0)
confusionMatrix(pred.rpart, validation$Survived)
accur.rpart <- confusionMatrix(pred.rpart, validation$Survived)$overall[[1]]

# Random Forest Model (Boosted Trees) - library(randomForest)
fit.rf <- randomForest(as.factor(Survived) ~., data = subTraining,
                       importance = T, ntree = 200)
pred.rf <- predict(fit.rf, validation)
confusionMatrix(pred.rf, validation$Survived)
accur.rf <- confusionMatrix(pred.rf, validation$Survived)$overall[[1]]

# cForest - library(party)
fit.cforest <- cforest(as.factor(Survived) ~., data = subTraining, controls = cforest_unbiased(ntree=2000, mtry=3)) 
pred.cforest <- predict(fit.cforest, validation, OOB = T, type = "response")
confusionMatrix(pred.cforest, validation$Survived)
accur.cforest <- confusionMatrix(pred.cforest, validation$Survived)$overall[[1]]

# Gradient Boosted Model - library(gbm)
fit.gbm <- gbm(Survived ~., data = subTraining, distribution = "bernoulli", shrinkage = 0.01,
               interaction.depth = 2, n.trees = 2000)
pred.gbm <- predict(fit.gbm, validation, n.trees = 2000, "response")

predGBM <- prediction(pred.gbm, validation$Survived)
roc.perfGBM = performance(predGBM, measure = "tpr", x.measure = "fpr")
plot(roc.perfGBM, col = "red", lwd = 2)
abline(a=0,b=1,lwd=2,lty=2,col="gray")

# Function for finding the optimal cutoff 
acc.perfGBM = performance(predGBM, measure = "acc")
plot(acc.perfGBM, col = "blue")
indGBM = which.max( slot(acc.perfGBM, "y.values")[[1]] )
accGBM = slot(acc.perfGBM, "y.values")[[1]][indGBM]
cutoffGBM = slot(acc.perfGBM, "x.values")[[1]][indGBM]
print(c(accuracy= accGBM, cutoff = cutoffGBM))
#auc.perfGBM = performance(predGBM, measure = "auc")
#print(auc.perfGBM)

pred.gbm <- ifelse(pred.gbm >= cutoffGBM, 1, 0)
confusionMatrix(pred.gbm, validation$Survived)
accur.gbm <- confusionMatrix(pred.gbm, validation$Survived)$overall[[1]]

# Naive Bayes - library(e1071)
fit.nb <- naiveBayes(as.factor(Survived) ~., data = subTraining, laplace = 3)
pred.nb <- predict(fit.nb, validation, type = "raw")
pred.nb <- pred.nb[, 2]

predNB <- prediction(pred.nb, validation$Survived)
roc.perfNB = performance(predNB, measure = "tpr", x.measure = "fpr")
plot(roc.perfNB, col = "red", lwd = 2)
abline(a=0,b=1,lwd=2,lty=2,col="gray")

# Function for finding the optimal cutoff 
acc.perfNB = performance(predNB, measure = "acc")
plot(acc.perfNB, col = "blue")
indNB = which.max( slot(acc.perfNB, "y.values")[[1]] )
accNB = slot(acc.perfNB, "y.values")[[1]][indNB]
cutoffNB = slot(acc.perfNB, "x.values")[[1]][indNB]
print(c(accuracy= accNB, cutoff = cutoffNB))
#auc.perfNB = performance(predNB, measure = "auc")
#print(auc.perfNB)

pred.nb <- ifelse(pred.nb >= cutoffNB, 1, 0)
confusionMatrix(pred.nb, validation$Survived)
accur.nb <- confusionMatrix(pred.nb, validation$Survived)$overall[[1]]

# Support Vector Machines - library(e1071)
fit.svm <- svm(as.factor(Survived) ~., data = subTraining, kernel = "radial",
               degree = 3, cross = 4)
pred.svm <- predict(fit.svm, validation)
confusionMatrix(pred.svm, validation$Survived)
accur.svm <- confusionMatrix(pred.svm, validation$Survived)$overall[[1]]

# Artificial Neural Network - library(nnet)
fit.nnet <- train(as.factor(Survived) ~., data = subTraining, method = "nnet", trace = F)
pred.nnet <- predict(fit.nnet, validation)
confusionMatrix(pred.nnet, validation$Survived)
accur.nnet <- confusionMatrix(pred.nnet, validation$Survived)$overall[[1]]

# Learning Vector Quantization 
grid <- expand.grid(size=c(5,10,20,50), k=c(1,2,3,4,5))
fit.lvq <- train(as.factor(Survived) ~., data = subTraining, method = "lvq", tuneGrid = grid)
pred.lvq <- predict(fit.lvq, validation)
confusionMatrix(pred.lvq, validation$Survived)
accur.lvq <- confusionMatrix(pred.lvq, validation$Survived)$overall[[1]]

# Flexible Discriminant Analysis 
fit.fda <- train(as.factor(Survived) ~., data = subTraining, method = "fda")
pred.fda <- predict(fit.fda, validation)
confusionMatrix(pred.fda, validation$Survived)
accur.fda <- confusionMatrix(pred.fda, validation$Survived)$overall[[1]]

# Multinomial Logit Model
fit.ml <- multinom(as.factor(Survived) ~., data = subTraining)
pred.ml <- predict(fit.ml, validation)
confusionMatrix(pred.ml, validation$Survived)
accur.ml <- confusionMatrix(pred.ml, validation$Survived)$overall[[1]]




Algorithm <- c("CART", "Random Forest", "cForest", "GBM", "Naive Bayes", "SVM",
               "ANN", "LVQ", "FDA", "Multinomial Logit Model")
Accuracy <- c(accur.rpart, accur.rf, accur.cforest, accur.gbm, accur.nb, accur.svm,
              accur.nnet, accur.lvq, accur.fda, accur.ml)
TopPerformance <- data.frame(Accuracy)
colnames(TopPerformance) <- "Accuracy"
rownames(TopPerformance) <- Algorithm


# Predicting and Writing the final prediction
predDF <- data.frame(CART = pred.rpart, RF = pred.rf, GBM = pred.gbm, NNET = pred.nnet, 
                     SVM = pred.svm, Survived = validation$Survived)
fit.final <- gbm(Survived ~., distribution = "bernoulli", data = predDF, shrinkage = 0.01,
                 interaction.depth = 2, n.trees = 2000)
pred.final <- predict(fit.final, predDF, n.trees = 2000, "response")

predGBM1 <- prediction(pred.final, validation$Survived)
roc.perfGBM1 = performance(predGBM1, measure = "tpr", x.measure = "fpr")
plot(roc.perfGBM1, col = "red", lwd = 2)
abline(a=0,b=1,lwd=2,lty=2,col="gray")

# Function for finding the optimal cutoff 
acc.perfGBM1 = performance(predGBM1, measure = "acc")
plot(acc.perfGBM1, col = "blue")
indGBM1 = which.max( slot(acc.perfGBM1, "y.values")[[1]] )
accGBM1 = slot(acc.perfGBM1, "y.values")[[1]][indGBM1]
cutoffGBM1 = slot(acc.perfGBM1, "x.values")[[1]][indGBM1]
print(c(accuracy= accGBM1, cutoff = cutoffGBM1))
#auc.perfGBM1 = performance(predGBM1, measure = "auc")
#print(auc.perfGBM1)

pred.final <- ifelse(pred.final >= cutoffGBM1, 1, 0)
confusionMatrix(pred.final, validation$Survived)


pred.rpartF <- predict(fit.rpart, testing)
pred.rpartF <- ifelse(pred.rpartF >= cutoffCART, 1, 0)
pred.rfF <- predict(fit.rf, testing)
pred.gbmF <- predict(fit.gbm, testing, n.trees = 2000, "response")
pred.gbmF <- ifelse(pred.gbmF >= cutoffGBM, 1, 0)
pred.nnetF <- predict(fit.nnet, testing)
pred.svmF <- predict(fit.svm, testing)

finalDF <- data.frame(CART = pred.rpartF, RF = pred.rfF, GBM = pred.gbmF, NNET = pred.nnetF, 
                     SVM = pred.svmF)

pred.submission <- predict(fit.final, finalDF, n.trees = 2000, "response")
pred.submission <- ifelse(pred.submission >= cutoffGBM1, 1, 0)
PassengerId <- 892:1309
submit <- data.frame(PassengerId = PassengerId, Survived = pred.submission)
write.csv(submit, file = "Ensemble9.csv", row.names = FALSE)


