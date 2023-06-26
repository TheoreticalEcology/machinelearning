trim_name = function(name) {
  name = stringr::str_replace_all( name , "'", " ")
  name = stringr::str_replace_all( name , '"', " ")
  name = stringr::str_remove_all(name, "-")
  return(name)
}

create_test = function(input_file) {
  name = strsplit(input_file, ".", fixed = TRUE)[[1]][1]
  name = strsplit(name, "/", fixed = TRUE)[[1]][2]
  output = output = paste0("tests/", name, ".R")
  knitr::purl(input_file, output = output, documentation = 1, quiet = TRUE)
  input = file(output)
  code = readLines(input)
  close(input)
  file.remove(output)
  result = ''
  open = FALSE
  end = c( "list2env(as.list(environment()), envir = .GlobalEnv)", "}, NA)})" )
  
  for(i in 1:length(code)){
    if(stringr::str_detect(code[i], "## ----")){
      if(open){
        result = c(result, end)
        result = c(result, paste0("testthat::test_that('", trim_name(code[i]),
                                  "', {testthat::expect_error( {"))
      }else{
        result = c(result, paste0("testthat::test_that('", trim_name(code[i]),
                                  "', {testthat::expect_error( {"))
        open = TRUE
      }
    }
    result = c(result, code[i])
  }
  
  result = c(result, end, "rm(list=ls())", "gc()")
  writeLines(result, con = paste0("tests/", "test-", name, ".R"))
}
