install.packages("xgboost")
library("xgboost")
setwd("A:/UNCC/Spring 2017/ML/Project/Clone/WithStopWordsRemoval")
QuoraData<-read.csv(file ="ToAshok1.csv")
QuoraData



names(trainData)
#dropping columns

QuoraData$Index<- NULL
QuoraData$Cosine_Stop<-NULL
QuoraData <- QuoraData[ -c(1,26) ]
QuoraData
names(QuoraData)


nrow(QuoraData)
smp_size <- floor(0.55 * nrow(QuoraData))
smp_size

index <- sample(1:nrow(QuoraData), size = smp_size)
index
trainData <- QuoraData[index, ]
testData <- QuoraData[-index, ]
dim(trainData)
dim(testData)



library("randomForest")
rf <- randomForest(trainData$is_duplicate ~ ., data = trainData,mtry = 2, importance = TRUE)
rf