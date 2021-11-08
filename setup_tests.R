source("tests/utils.R")

#### INFO #####
# All chunks will be tested within a testthat::test_that environment.
# To skip chunks, use purl=FALSE in chunk options.
# For successful debugging, please name all chunks. 
# Chunk names+arguments will be used as names for tests.

# 1. Specify Rmds to test:
rmds = c("book/index.Rmd",
         "book/02_R-reminder.Rmd",
         "book/03-intro.Rmd",
         "book/04-fundamental.Rmd",
         "book/05-Deep.Rmd",
         "book/06-xAI.Rmd",
         "book/07-GAN.Rmd",
         "book/08-Datasets.Rmd"
)[c(1, 4, 5, 6, 8)]

# 2. Create tests
.n = sapply(rmds, create_test)
Sys.sleep(5)

# 3. Run tests
testthat::test_dir("tests/")

# 4. Clean/delete tests
files = list.files("tests/", pattern = "test-*", full.names = TRUE)
file.remove(files)
