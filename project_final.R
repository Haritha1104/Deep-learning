library(keras)
library(dplyr)
library(tensorflow)
library(reticulate)
library(tfruns)
library(factoextra)
library(FactoMineR)
tensorflow::tf$random$set_seed(1)
load("data_activity_recognition.RData")

#Data preparation
x_train = array_reshape(x_train,c(nrow(x_train), 125*45))
x_test  = array_reshape(x_test, c(nrow(x_test) , 125*45))
dim(x_train)
dim(x_test)

#Normalize the input data
range_norm <- function(x, a = 0, b = 1) {
  ( (x - min(x)) / (max(x) - min(x)) )*(b - a) + a }

x_train <- apply(x_train, 2, range_norm) 
x_test  <- apply(x_test,  2, range_norm) 
range(x_train)
range(x_test)

#Convert y_train and y_test to one-hot coding  
y <- as.factor(y_train)
y_train <- factor(y , levels = levels(y) , labels = 0:18) 
y_train <- to_categorical(y_train)  
dim(y_train)

y2 <- as.factor(y_test)
y_test <- factor(y2, levels = levels(y2) , labels = 0:18) 
y_test <- to_categorical(y_test) 
dim(y_test)

##PCA for dimension Reduction for x_train and x_test
pca <- prcomp(x_train, scale. = TRUE)
dim(pca$x)

#standard deviation of each principal component
sd <- pca$sdev
#variance
var <- std_dev^2
#Proportion of variance explained
p <- pr_var/sum(pr_var)
#Cumulative scree plot
plot(cumsum(p), xlab = "Principal Component",
     ylab = "Cumulative Proportion of Variance Explained",
     type = "b")

x_train <- data.frame(pca$x)
x_train <- x_train[,1:500] #since 500 is where elbow appears

#Transform the test data into PCA
test.data <- predict(pca, newdata = x_test)
test.data <- as.data.frame(test.data)

#Select the first 500 components
x_test <- test.data[,1:500]

#Converting data frames to matrices
x_train <- as.matrix(x_train)
x_test <- as.matrix(x_test)
str(x_train)
str(x_test)

#split the test data into test and validation
val <- sample(1:nrow(x_test), 760)
test <- setdiff(1:nrow(x_test), val)
x_val <- x_test[val,]
y_val <- y_test[val,]
x_test <- x_test[test,]
y_test <- y_test[test,]
dim(x_val)
dim(y_val)
dim(x_test)
dim(y_test)

#model configuration for 2 layers
model <- keras_model_sequential()
model %>%
  layer_dense(units = 300, activation = "relu", input_shape = ncol(x_train)) %>%
  layer_dense(units = 150, activation = "relu") %>%
  layer_dense(units = 19, activation = "softmax") %>%
  compile(loss = "categorical_crossentropy", metrics = "accuracy",
          optimizer = optimizer_sgd(),)

summary(model)

#Fitting the model
fit <- model %>%
  fit(x_train, y_train,
      epochs = 100,
      batch_size = 32,
      validation_split = 0.2)

#Evaluating model with test data
evaluate(model, x_test, y_test)$accuracy

#Adding a smooth line to points
smooth_line <- function(y) {
  x <- 1:length(y)
  out <- predict(loess(y ~ x))
  return(out)
}

cols <- c("black","darkblue","gray","deepskyblue")

#Checking performance
out <- 1-cbind(fit$metrics$accuracy, fit$metrics$val_accuracy)
matplot(out, pch = 19, ylab = "Error", xlab = "Epochs",
        col = adjustcolor(cols[1:2], 0.3), log = "y")
matlines(apply(out, 2, smooth_line), lty = 1, col = cols[1:2], lwd = 2)
legend("topright", legend = c("Training", "Test"),
       fill = cols[1:2], bty = "n")

#minimum train and validation errors
apply(out, 2, min)


#model configuration for three layers
model2 <- keras_model_sequential()
model2 %>%
  layer_dense(units = 300, activation = "relu", input_shape = ncol(x_train)) %>%
  layer_dense(units = 150, activation = "relu") %>%
  layer_dense(units = 70, activation = "relu") %>%
  layer_dense(units = 19, activation = "softmax") %>%
  compile(loss = "categorical_crossentropy", metrics = "accuracy",
          optimizer = optimizer_sgd(),)

#Fitting the model
fit2 <- model2 %>%
  fit(x_train, y_train,
      epochs = 100,
      batch_size = 32,
      validation_split=0.2)

#Evaluating model with test data
evaluate(model2, x_test, y_test)$accuracy

cols <- c("black","darkblue","gray","deepskyblue")

#Checking performance
out <- 1-cbind(fit2$metrics$accuracy, fit2$metrics$val_accuracy)
matplot(out, pch = 19, ylab = "Error", xlab = "Epochs",
        col = adjustcolor(cols[1:2], 0.3), log = "y")

matlines(apply(out, 2, smooth_line), lty = 1, col = cols[1:2], lwd = 2)
legend("topright", legend = c("Training", "Test"),
       fill = cols[1:2], bty = "n")

#minimum train and validation errors
apply(out, 2, min)

#Checking performance of both models
cols2<- c("red" , "dodgerblue3")

out <- cbind(fit$metrics$val_accuracy, fit2$metrics$val_accuracy)
matplot(out, pch = 19, ylab = "Accuracy", xlab = "Epochs",
        col = adjustcolor(cols2[1:2], 0.3), log = "y")

matlines(apply(out, 2, smooth_line), lty = 1, col = cols2[1:2], lwd = 2)
legend("topright", legend = c("2 layers", "3 layers"),
       fill = cols2[1:2], bty = "n")

#Tuning different configurations
N <- nrow(x_train)

runs <- tuning_run("project_conf_1.R",
                   runs_dir = "runs_project1", 
                   flags = list(
                     dense_units1 =c(300,150),
                     dense_units2 =c(150,70),
                     dropout = c(0, 0.4, 0.5),
                     lr=c(0.001, 0.005, 0.01),
                     bs=c(0.005, 0.01, 0.03)*N
                   ),
                   sample = 0.1)


runs2 <- tuning_run("project_conf_2.R",
                    runs_dir = "runs_project2", 
                    flags = list(
                      dense_units1 =c(300,150),
                      dropout = c(0, 0.4, 0.5),
                      dense_units2 =c(150,100),
                      dense_units3 = c(100,50),
                      lr =c(0.001, 0.005, 0.01),
                      bs=c(0.005, 0.01, 0.03)*N
                    ),
                    sample = 0.1)


