
library(reticulate)
library(tensorflow)
install_tensorflow()
library(keras)
install_keras()
library(EBImage)
library(stringr)
library(pbapply)

# clear the table
rm(list=ls())

# test to make sure images are importing
cleaver <- readImage("lithic_train/cleaver.1085f.jpg")
display(cleaver)

# Set image size, reducing size will help process images faster
width <- 50
height <- 37.5

extract_feature <- function(dir_path, width, height, labelsExist = T) {
  img_size <- width * height
  
  ## List images in path
  images_names <- list.files(dir_path)
  
  if(labelsExist){
    ## Select only cleaver or handaxe images
    cleaverhandaxe <- str_extract(images_names, "^(cleaver|handaxe)")
    # Set cleaver == 0 and handaxe == 1
    key <- c("cleaver" = 0, "handaxe" = 1)
    y <- key[cleaverhandaxe]
  }
  
  print(paste("Start processing", length(images_names), "images"))
  ## Turn image into greyscale
  feature_list <- pblapply(images_names, function(imgname) {
    ## Read image
    img <- readImage(file.path(dir_path, imgname))
    ## Resize image
    img_resized <- resize(img, w = width, h = height)
    ## Set to grayscale (normalized to max)
    grayimg <- channel(img_resized, "gray")
    ## Get the image as a matrix
    img_matrix <- grayimg@.Data
    ## Coerce to a vector (row-wise)
    img_vector <- as.vector(t(img_matrix))
    return(img_vector)
  })
  ## bind the list of vector into matrix
  feature_matrix <- do.call(rbind, feature_list)
  feature_matrix <- as.data.frame(feature_matrix)
  ## Set names
  names(feature_matrix) <- paste0("pixel", c(1:img_size))
  
  if(labelsExist){
    return(list(X = feature_matrix, y = y))
  }else{
    return(feature_matrix)
  }
}

# Takes approx. 5min
trainData <- extract_feature("lithic_train/", width, height)
# 3 minutes
testData <- extract_feature("lithic_test/", width, height, labelsExist = F)


# Check processing 
par(mar = rep(0, 4))
test_cleaver <- t(matrix(as.numeric(trainData$X[2,]),
                    nrow = width, ncol = height, T))
image(t(apply(test_cleaver, 2, rev)), col = gray.colors(12),
      axes = F)

# Files are large, best to save them
save(trainData, testData, file = "cleaverhandaxe.RData")
load(file = "cleaverhandaxe.RData")

# Fix structure for 2d CNN
train_array <- layer_dropout(rate = 0.25) %>%
  layer_flatten() %>%
  layer_dense(units = 50, activation = "relu") %>%
  layer_dropout(rate = 0.25) %>%
  layer_dense(units = 1, activation = "sigmoid")

summary(model)

model %>% compile(
  loss = 'binary_crossentropy',
  optimizer = "adam",
  metrics = c('accuracy')
)

history %>%  fit(
  x = train_array, y = as.numeric(trainData$y),
  epochs = 30, batch_size = 100,
  validation_split = 0.2
)

plot(history)
