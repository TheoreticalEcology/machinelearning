--- 
title: "Machine Learning and AI in TensorFlow and R"
author: "Maximilian Pichler and Florian Hartig"
date: "2022-07-08"
site: bookdown::bookdown_site
output: 
  bookdown::gitbook:
    highlight: kate
documentclass: book
bibliography: ["packages.bib", "literature.bib"]
biblio-style: "apalike"
link-citations: yes
github-repo: rstudio/bookdown-demo
description: "Machine Learning and AI in TensorFlow and R"
---

# Prerequisites {#prerequisites}

```{=html}
<!-- Put this here (right after the first markdown headline) and only here for each document! -->
<script src="./scripts/multipleChoice.js"></script>
```




## R System

Make sure you have a recent version of R (>=3.6, ideally >=4.0) and RStudio on your computers. 


## TensorFlow and Keras

If you want to run the code on your own computers, you also will need to install TensorFlow / Keras for R. For this, the following should work for most people.

Run in R: 


```r
install.packages("keras", dependencies = T)
keras::install_keras()
```

This should work on most computers, in particular if all software is recent. Sometimes, however, things don't work well, especially the python distribution often makes problems. If the installation does not work for you, we can look at it together. Also, we will provide some virtual machines in case your computers / laptops are too old or you don't manage to install TensorFlow.

**Warning**: *You need at least TensorFlow version 2.6, otherwise, the argument "learning_rate" must be "lr"!*


## Torch for R

We may also use Torch for R. This is an R frontend for the popular PyTorch framework. To install Torch, type in R:


```r
install.packages("torch")
library(torch)
torch::install_torch()
```


## EcoData

We may sometimes use data sets from the EcoData package. To install the package, run:


```r
devtools::install_github(repo = "TheoreticalEcology/EcoData", 
                         dependencies = TRUE, build_vignettes = TRUE)
```

The default installation will install a number of packages that are useful for statistics. Especially in Linux, this may take some time to install. If you are in a hurry and only want the data, you can also run


```r
devtools::install_github(repo = "TheoreticalEcology/EcoData", 
                         dependencies = FALSE, build_vignettes = FALSE)
```

## Further Used Libraries

We will make huge use of different libraries. So take a coffee or two (that will take a while...) and install the following libraries.
Please do this in the given order unless you know what you're doing, because there are some dependencies between the packages.


```r
install.packages("abind")
install.packages("animation")
install.packages("ape")
install.packages("BiocManager")
BiocManager::install(c("Rgraphviz", "graph", "RBGL"))
install.packages("coro")
install.packages("dbscan")
install.packages("dendextend")
install.packages("devtools")
install.packages("dplyr")
install.packages("e1071")
install.packages("factoextra")
install.packages("fields")
install.packages("forcats")
install.packages("glmnet")
install.packages("gym")
install.packages("kknn")
install.packages("knitr")
install.packages("iml")
install.packages("lavaan")
install.packages("lmtest")
install.packages("magick")
install.packages("mclust")
install.packages("Metrics")
install.packages("microbenchmark")
install.packages("missRanger")
install.packages("mlbench")
install.packages("mlr3")
install.packages("mlr3learners")
install.packages("mlr3measures")
install.packages("mlr3pipelines")
install.packages("mlr3tuning")
install.packages("paradox")
install.packages("partykit")
install.packages("pcalg")
install.packages("piecewiseSEM")
install.packages("purrr")
install.packages("randomForest")
install.packages("ranger")
install.packages("rpart")
install.packages("rpart.plot")
install.packages("scales")
install.packages("semPlot")
install.packages("stringr")
install.packages("tfprobability")
install.packages("tidyverse")
install.packages("torchvision")
install.packages("xgboost")

devtools::install_github("andrie/deepviz", dependencies = TRUE,
                         upgrade = "always")
devtools::install_github('skinner927/reprtree')
devtools::install_version("lavaanPlot", version = "0.6.0")

reticulate::conda_install("r-reticulate", packages = "scipy", pip = TRUE)

```


## Linux/UNIX systems have to fulfill some further dependencies

**Debian based systems**

For Debian based systems, we need:

```{}
build-essential
gfortran
libmagick++-dev
r-base-dev
```

If you are new to installing packages on Debian / Ubuntu, etc., type the following:

```{}
sudo apt update && sudo apt install -y --install-recommends build-essential gfortran libmagick++-dev r-base-dev
```

## Reminders About Basic Operations in R 

Basic and advanced knowledge of R is required to successfully participate in this course. If you would like to refresh your knowledge of R, you can review the chapter ['Reminder: R Basics'](https://theoreticalecology.github.io/AdvancedBiostatistics/reminder.html) from the advanced statistic course.  

**Authors**:

Maximilian Pichler: [\@_Max_Pichler](https://twitter.com/_Max_Pichler)

Florian Hartig: [\@florianhartig](https://twitter.com/florianhartig)


**Contributors**:

Johannes Oberpriller, Matthias Meier



```{=html}
<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>
```
