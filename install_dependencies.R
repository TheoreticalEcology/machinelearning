source("book/dependencies.R")
install.packages("keras")
# It's annoying to setup keras 
try({ 
  reticulate::install_miniconda()
  keras::install_keras() 
  reticulate::use_condaenv("r-reticulate")
  }, silent = TRUE)
install.packages("torch")
torch::install_torch()
