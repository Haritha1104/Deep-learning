
#Haritha Ramachandran

library(tfruns)


FLAGS <- flags(
  flag_numeric("dropout", 0.4),
  flag_numeric("lr", 0.01),
  flag_numeric("bs", 100),
  flag_integer("dense_units1",256),
  flag_integer("dense_units2",128),
  flag_integer("dense_units3",64)
)

# model configuration



model <- keras_model_sequential() %>%
  
  layer_dense(units = FLAGS$dense_units1, input_shape = ncol(x_train), activation = "relu", name = "layer_1",
              kernel_regularizer = regularizer_l2(0.01)) %>%
  layer_dropout(rate = FLAGS$dropout) %>%
  layer_dense(units = FLAGS$dense_units2, activation = "relu", name = "layer_2",
              kernel_regularizer = regularizer_l2(0.01)) %>%
  layer_dropout(rate = FLAGS$dropout) %>%
  layer_dense(units = FLAGS$dense_units3, activation = "relu", name = "layer_3",
              kernel_regularizer = regularizer_l2(0.01)) %>%
  layer_dropout(rate = FLAGS$dropout) %>%
  layer_flatten()%>%
  layer_dense(units = 19, activation = "softmax", name = "layer_out") %>%
  compile(loss = "categorical_crossentropy", metrics = "accuracy",
          optimizer = optimizer_adam(lr = FLAGS$lr),)

fit <- model %>% fit(
  x = x_train, y = y_train,
  validation_data = list(x_val, y_val),
  epochs = 50,
  batch_size = FLAGS$bs,
  verbose = 1,
  callbacks = callback_early_stopping(monitor = "val_accuracy", patience = 20))

# store accuracy on test set for each run
score <- model %>% evaluate(
  x_test, y_test,
  verbose = 0)




read_metrics <-function(path,files =NULL)# 'path' is where the runs are --> e.g. "path/to/runs"
{
  path <-paste0(path,"/")
  if(is.null(files) ) files <-list.files(path)
  n <-length(files)
  out <-vector("list", n)
  for( i in 1:n ) {
    dir <-paste0(path, files[i],"/tfruns.d/")
    out[[i]] <-jsonlite::fromJSON(paste0(dir,"metrics.json"))
    out[[i]]$flags <-jsonlite::fromJSON(paste0(dir,"flags.json"))
    #out[[i]]$evaluation <-jsonlite::fromJSON(paste0(dir,"evaluation.json"))
  }
  return(out)
}


plot_learning_curve <-function(x,ylab =NULL,cols =NULL,top =3,span =0.4, ...)
{# to add a smooth line to points
  smooth_line <-function(y) {
    x <-1:length(y)
    out <-predict(loess(y~x,span =span) )
    return(out)
  }
  matplot(x,ylab =ylab,xlab ="Epochs",type ="n", ...)
  grid()
  matplot(x,pch =19,col =adjustcolor(cols,0.3),add =TRUE)
  tmp <-apply(x,2, smooth_line)
  tmp <-sapply( tmp,"length<-",max(lengths(tmp)) )
  set <-order(apply(tmp,2, max,na.rm =TRUE),decreasing =TRUE)[1:top]
  cl <-rep(cols,ncol(tmp))
  cl[set] <-"deepskyblue2"
  matlines(tmp,lty =1,col =cl,lwd =2)
}



# extract results
out <-read_metrics("runs_project_fin22")
# extract validation accuracy and plot learning curve
acc <-sapply(out,"[[","val_accuracy")
plot_learning_curve(acc,col =adjustcolor("black",0.3),ylim = c(0.10, 1.0),ylab ="Val accuracy",top =4)

res <- ls_runs(metric_val_accuracy > 0.85, runs_dir = "runs_project_fin22", order=metric_val_accuracy)
res <- res[, c(2,4,8:13)]
res[1:4,]



