library(tensorflow)
tf$enable_eager_execution()

preprocess = function(x){
  image = tf$io$read_file(x)
  image = tf$io$decode_jpeg(image, channels=3)
  image = tf$image$resize(image, c(80L, 80L))
  image
}
dirs = list.dirs("../../../Documents/Flower/flowers",full.names = TRUE)

files = lapply(dirs[-1], function(d) list.files(d, full.names = TRUE))

images = lapply(files, function(f) tf$map_fn(preprocess, tf$constant(f), tf$float32))
names = c("daisy", "dandelion", "rose", "sunflower", "tulip")
set.seed(42)
for(i in 1:length(images)){
  sp = images[[i]]
  dd = sp$shape$as_list()[1]
  indices = sample.int(dd, 0.7*dd)
  t_indices = (1:dd)[!1:dd %in% indices]
  train = tf$gather(sp, indices-1L, axis = 0L)
  test = tf$gather(sp, t_indices-1L, axis = 0L)
  saveRDS(train$numpy(), file = paste0("Day3/flower/train_",names[i], "_.RDS"), version = 2)
  saveRDS(test$numpy(), file = paste0("Day3/flower/test_",names[i], "_.RDS"), version = 2)
}
