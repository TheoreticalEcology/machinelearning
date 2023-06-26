
#' Build a questionnaire tag
#'
#' Build the questionnaire tag, ready to be inserted in a HTML document.
makeQuestionnaireTag = function(){
  print("As R is bad, it terminates program execution when you press ESC. so don't press ESC...")

  repeat{
    question = readline(prompt = "Please enter question: ")
    if(question != "") break
  }

  repeat{
    kind = readline(prompt = "Specify kind of question (0: text, 1: radiobutton, 2: checkbox): ")
    if(kind %in% c(0, 1, 2, "text", "radio", "radiobutton", "radiobutton", "checkbox", "checkboxes")){ break }
  }
  if(kind %in% c(0, "text")){ kind = "text" }
  else if(kind %in% c(1, "radio", "radiobutton", "radiobuttons)")){ kind = "radio" }
  else if(kind %in% c(2, "checkbox", "checkboxes")){ kind = "checkbox" }

  comment = readline(prompt = "Any general comment for this question: ")

  answers = list()
  counter = 0
  repeat{
      if(counter){
        answer = ""
        answer = readline(prompt = "Specify a further answer (leave empty if no more answers should be entered): ")
        if(answer == ""){ break }
      }else{
        repeat{
          answer = readline(prompt = "Specify an answer: ")
          if(answer != ""){ break }
        }
      }

      counter = counter + 1

      repeat{
        correct = readline(prompt = paste("Is this answer [", answer, "] correct (true/false)?: ", sep = ""))
        if(correct %in% c(0, "f", "F", "false", "False", "FALSE", 1, "t", "T", "true", "True", "TRUE")){ break }
      }
      if(correct %in% c(0, "F", "false", "False", "FALSE")){ correct = "false" }else{ correct = "true" }

      explanationIfSelected = readline(prompt = "Any explanation if selected?: ")
      explanationIfNotSelected = readline(prompt = "Any explanation if not selected?: ")
      explanationGeneral = readline(prompt = "Any general explanation?: ")

      answers[counter] = paste("\n\t\t\t{\n\t\t\t\t'answer':'", answer, "',\n\t\t\t\t'correct':", correct, ",\n\t\t\t\t'explanationIfSelected':'",
                               explanationIfSelected, "',\n\t\t\t\t'explanationIfNotSelected':'",
                               explanationIfNotSelected, "',\n\t\t\t\t'explanationGeneral':'", explanationGeneral, "'\n\t\t\t}", sep = "")
  }

  tag = paste("<script>\n\tmakeMultipleChoiceForm(\n\t\t'", question, "',\n\t\t'", kind, "',\n\t\t[,", sep = "")
  if(counter > 1){ for(i in 1:(counter - 1)){ tag = paste(tag, answers[i], ",", sep = "") } }   #All but the last get a comma separator.
  tag = paste(tag, answers[counter], "\n\t\t],\n\t\t'", comment, "'\n\t);\n</script>", sep = "")

  return(tag)
}



