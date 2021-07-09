/*
*	This script is for generating multiple choice questions.
*/


/**********		Examples	**********
<script>
	makeMultipleChoiceForm(
		'How much is the fish?',
		'text',
			[
				{
					'answer':'123',
					'correct':true,
					'explanationIfSelected':'Right guess.',
					'explanationIfNotSelected':'Wrong guess.',
					'explanationGeneral':''
				},
				{
					'answer':'321',
					'correct':false,
					'explanationIfSelected':'Wrong guess.',
					'explanationIfNotSelected':'',
					'explanationGeneral':''
				}
			],
		'comment text'
	);

</script>

<script>
	makeMultipleChoiceForm(
		'How much is the fish?',
		'radio',
			[
				{
					'answer':'123',
					'correct':true,
					'explanationIfSelected':'Right guess.',
					'explanationIfNotSelected':'This would have been a right number.',
					'explanationGeneral':''
				},
				{
					'answer':'321',
					'correct':true,
					'explanationIfSelected':'2nd right guess.',
					'explanationIfNotSelected':'This would have been a right number.',
					'explanationGeneral':''
				}
			],
		'comment radio'
	);
</script>


<script>
	makeMultipleChoiceForm(
		'How much is the fish?',
		'checkbox',
			[
				{
					'answer':'123',
					'correct':true,
					'explanationIfSelected':'Right guess.',
					'explanationIfNotSelected':'This would have been the right number.',

					'explanationGeneral':''
				},
				{
					'answer':'321',
					'correct':false,

					'explanationIfSelected':'Wrong guess.',
					'explanationIfNotSelected':'Well done.',
					'explanationGeneral':''
				}

			],
		'comment checkbox'
	);
</script>

/**********		Examples end	**********







/*
*	Naming convention for entities:
*		General:
*			Every question is named with the type ('radio', 'checkbox' or 'test') and (separated with '_') an ongoing ID number (uniqueSelectObjectID).
*			The purpose of dollar signs is just to distinguish between several choosable answer alternatives.
*
*		Radiobuttons and checkboxes:
*			Every answer is divided from the question ID with a '$' and gets a local ongoing ID. For example 'radio_0$0' and 'radio_0$1'.
*	
*		Textfields:
*			The textfield has a local ID '0', separated with '$'. For example 'text_1$0'.
*
*		Explanations:
*			Explanation fields (answer specific) are named with the respective explanation kind
*			('explanationIfSelected', 'explanationIfNotSelected', 'explanationGeneral'), separated with '_' from the respective answer.
*			This holds for all types ('radio', 'checkbox' and 'test'). For example 'text_1$2_explanationGeneral' or 'radio_0$5_explanationIfSelected'.
*
*			Textfields might have several answers with specific reactions. For example 'text_1$0_explanationGeneral' or 'text_1$1_explanationGeneral'.
*			These explanation fields need to be distinguishable, but no answer fields exist.
*
*			Comment fields (question specific) have the respective object name and '_comment' after. For example 'text_1_comment'.
*
*		Indicator of correctness is named like the respective object, followed by "_correctness". For example 'checkbox_2_correctness'.
*	
*/



var uniqueSelectObjectID = 0;		//!	Global(!) counter for inserted multiple choice forms.
var correctAnswers = {};			//Global variable for storing (indices of) correct answers.
var textAnswers = {};				//Global variable for storing all supplied text answers.


/*!
*	\brief	Is called when a submit button was clicked.
*			Evaluates the given answers and makes the corresponding explanation and comment fields visible.
*			Makes use of the global variables "correctAnswers" and "textAnswers".
*	
*	@param	objectID[in]			Global question ID with kind prefix (kind_id).
*/
function evaluateQuestion(objectID){
	let splitted = objectID.split('_');
	let kind = splitted[0];
	let uniqueID = splitted[1];
	
	var correctAnswerIndices = correctAnswers[uniqueID];
	let userWasRight = false;
	
	if(kind == 'text'){
		let suppliedTextAnswers = textAnswers[uniqueID];
		let userInput = document.getElementById(objectID + '$0').value;
		
		for(let i = 0; i < suppliedTextAnswers.length; ++i){
			if(userInput == suppliedTextAnswers[i]){
				document.getElementById(objectID + '$' + i + '_explanationIfSelected').style.display = 'block';
				if(i in correctAnswerIndices){ userWasRight = true; }
			}else{
				document.getElementById(objectID + '$' + i + '_explanationIfNotSelected').style.display = 'block';
			}
			document.getElementById(objectID + '$' + i + '_explanationGeneral').style.display = 'block';
		}
	}else if(kind == 'radio'){
		let i = 0;
		while(element = document.getElementById(objectID + '$' + i)){
			if(element.checked){
				if(i in correctAnswerIndices){ userWasRight = true; }	//1 correct answer is enough.
				document.getElementById(objectID + '$' + i + '_explanationIfSelected').style.display = 'block';
			}else{
				document.getElementById(objectID + '$' + i + '_explanationIfNotSelected').style.display = 'block';
			}
			document.getElementById(objectID + '$' + i + '_explanationGeneral').style.display = 'block';
			++i;
		}
	}else if(kind == 'checkbox'){
		userWasRight = true;
		let i = 0;
		while(element = document.getElementById(objectID + '$' + i)){
			if(element.checked){
				if(!(i in correctAnswerIndices)){ userWasRight = false; }	//All anserws must be correct.
				document.getElementById(objectID + '$' + i + '_explanationIfSelected').style.display = 'block';
			}else{
				if(i in correctAnswerIndices){ userWasRight = false; }	//All anserws must be correct.
				document.getElementById(objectID + '$' + i + '_explanationIfNotSelected').style.display = 'block';
			}
			document.getElementById(objectID + '$' + i + '_explanationGeneral').style.display = 'block';
			++i;
		}
	}
	document.getElementById(objectID + '_correctness').innerText = 'Your answer was ' + (userWasRight ? 'correct.' : 'wrong.');
	document.getElementById(objectID + '_correctness').style.display = 'block';
	document.getElementById(kind + '_' + uniqueID + '_comment').style.display = 'block';
}


/*!
*	\brief	Generates a new (multiple choice) question form.
*			Just use this function in a script tag.
*	
*	The kind or the question could be 'radio', 'checkbox' or 'text'.
*	For radios, there could be more correct answers and the question is evaluated as correct, if one of the correct answers is given.
*	Checkboxes could have several correct answers and all (and no other) must be present, if the question should be evaluated as correct.
*	In both cases, no correct answer is allowed as well (empty array).
*	Questions with textboxes are evaluated as correct, if one of the supplied answers is given.
*	
*	@param	question[in]		A string that holds the question.
*	@param	kind[in]			A string that holds either 'radio', 'checkbox' or 'text'.
*	@param	answers[in]			An array of dictionaries. One dictionary for each answer.
*								Each dictionary must consist of the following key/value pairs:
*								'answer' -> Holds a string.
*								'correct' -> Must be be true or false.
*								'explanationIfSelected' -> Holds a string (answer specific). May be empty.
*								'explanationIfNotSelected' -> Holds a string (answer specific). May be empty.
*								'explanationGeneral' -> Holds a string (answer specific). May be empty.
*								Example:
*								[
*									{
*										'answer':'123',
*										'correct':true,
*										'explanationIfSelected':'Right guess.',
*										'explanationIfNotSelected':'This was the right number.',
*										'explanationGeneral':'You had to guess a number.'
*									},
*									{
*										'answer':'321',
*										'correct':false,
*										'explanationIfSelected':'Wrong guess.',
*										'explanationIfNotSelected':'Well done.',
*										'explanationGeneral':'You had to guess a number.'
*									}
*								]
*	@param	comment[in]			A string for a question specific explanation. May be empty.
*								Best choice for free text input, as not every possible answer could be catched.
*/
function makeMultipleChoiceForm(
	question,
	kind,
	answers,
	comment = ''
){
	if(kind != 'radio' && kind != 'checkbox' && kind != 'text'){return;}
	//Omit further type checking as I assume it would not be neccessary.
	
	
	//Locate position and insert new div as container for question and answers.
	var arrScripts = document.getElementsByTagName('script');	//Holds all script tags until now.
	var currScript = arrScripts[arrScripts.length - 1];			//Holds the current script tag.
	var thisDiv = document.createElement('div');	//Unnamed.
	currScript.parentNode.appendChild(thisDiv);	
	
	
	var questionHeading = document.createElement('H2');
	var questionText = document.createTextNode(question);
	questionHeading.appendChild(questionText);
	thisDiv.appendChild(questionHeading);
	
	let answerArray = [];
	
	if(kind == 'text'){
		//Unique identifier of the textfiled in the whole site.
			let identifier = kind + '_' + uniqueSelectObjectID + '$';	
		
		var textbox = document.createElement('input');
		textbox.type = 'text';
		textbox.id = identifier + '0';
		textbox.name = uniqueSelectObjectID;
		textbox.value = '';

		var newline = document.createElement('br');
		
		thisDiv.appendChild(textbox);
		thisDiv.appendChild(newline);
		
		let localElementCounter = 0;	//Holds the local counter for uniquely identifying answers.
		let textArray = [];
		
		//Traverse just to enable an explanation for every possible (given) answer, if needed.
		answers.map(function(element){	//Traverse all answers.
			var explanation = document.createElement('p');
			var text = document.createTextNode(element['explanationIfSelected']);
			explanation.appendChild(text);
			explanation.id = identifier + localElementCounter + '_explanationIfSelected';
			explanation.style.display = 'none';		//Invisible.
			//explanation.style.display = 'block';	//Visible.
			thisDiv.appendChild(explanation);
			
			var explanation = document.createElement('p');
			var text = document.createTextNode(element['explanationIfNotSelected']);
			explanation.appendChild(text);
			explanation.id = identifier + localElementCounter + '_explanationIfNotSelected';
			explanation.style.display = 'none';		//Invisible.
			//explanation.style.display = 'block';	//Visible.
			thisDiv.appendChild(explanation);
			
			var explanation = document.createElement('p');
			var text = document.createTextNode(element['explanationGeneral']);
			explanation.appendChild(text);
			explanation.id = identifier + localElementCounter + '_explanationGeneral';
			explanation.style.display = 'none';		//Invisible.
			//explanation.style.display = 'block';	//Visible.
			thisDiv.appendChild(explanation);
			
			
			//Indices of the explanation fields follow the normal numbering, starting at 0.
			textArray.push(element['answer']);
			if(element['correct']){ answerArray.push(localElementCounter); }
			
			++localElementCounter;
		});
		
		var newline = document.createElement('br');
		thisDiv.appendChild(newline);
		
		
		textAnswers[uniqueSelectObjectID] = textArray;
		
		//Wrapping explanations in an extra function makes nothing easier. It just causes chaos and overhead.
	}else{
		let localElementCounter = 0;	//Holds the local counter for uniquely identifying answers.
	
		answers.map(function(element){	//Traverse all answers.
			//Unique identifier of the answer in the whole site.
			let identifier = kind + '_' + uniqueSelectObjectID + '$' + localElementCounter;	
			
			var selectbox = document.createElement('input');
			selectbox.type = kind;
			selectbox.id = identifier;
			selectbox.name = uniqueSelectObjectID;	//Must correspond to the name of the other answers.
			selectbox.value = identifier;
			
			var label = document.createElement('label');
			label.htmlFor = identifier;
			
			var description = document.createTextNode(element['answer']);
			label.appendChild(description);
			
			var newline = document.createElement('br');
			
			thisDiv.appendChild(selectbox);
			thisDiv.appendChild(label);
			thisDiv.appendChild(newline);
			
			
			var explanation = document.createElement('p');
			var text = document.createTextNode(element['explanationIfSelected']);
			explanation.appendChild(text);
			explanation.id = identifier + '_explanationIfSelected';
			explanation.style.display = 'none';		//Invisible.
			//explanation.style.display = 'block';	//Visible.
			thisDiv.appendChild(explanation);
			
			var explanation = document.createElement('p');
			var text = document.createTextNode(element['explanationIfNotSelected']);
			explanation.appendChild(text);
			explanation.id = identifier + '_explanationIfNotSelected';
			explanation.style.display = 'none';		//Invisible.
			//explanation.style.display = 'block';	//Visible.
			thisDiv.appendChild(explanation);
			
			var explanation = document.createElement('p');
			var text = document.createTextNode(element['explanationGeneral']);
			explanation.appendChild(text);
			explanation.id = identifier + '_explanationGeneral';
			explanation.style.display = 'none';		//Invisible.
			//explanation.style.display = 'block';	//Visible.
			thisDiv.appendChild(explanation);
			
			var newline = document.createElement('br');
			thisDiv.appendChild(newline);
			
			
			if(element['correct']){ answerArray.push(localElementCounter); }
			
			++localElementCounter;
		});
	}
	var explanation = document.createElement('p');
	var text = document.createTextNode(comment);
	explanation.appendChild(text);
	explanation.id = kind + '_' + uniqueSelectObjectID + '_comment';
	explanation.style.display = 'none';		//Invisible.
	//explanation.style.display = 'block';	//Visible.
	thisDiv.appendChild(explanation);
	
	
	var newline = document.createElement('br');
	thisDiv.appendChild(newline);
	
	var submitButton = document.createElement('button');
	submitButton.id = kind + '_' + uniqueSelectObjectID;
	submitButton.innerHTML = 'Submit answer';
	submitButton.addEventListener('click', function(){
		this.disabled = true;
		evaluateQuestion(this.id);
	});
	thisDiv.appendChild(submitButton);
	
	
	var explanation = document.createElement('p');
	var text = document.createTextNode('');		//Change with document.getElementById(...).innerText = '...';
	explanation.appendChild(text);
	explanation.id = kind + '_' + uniqueSelectObjectID + '_correctness';
	explanation.style.display = 'none';		//Invisible.
	//explanation.style.display = 'block';	//Visible.
	thisDiv.appendChild(explanation);
	
	
	var newline = document.createElement('br');
	thisDiv.appendChild(newline);
	var newline = document.createElement('br');
	thisDiv.appendChild(newline);
	
	correctAnswers[uniqueSelectObjectID] = answerArray;
	
	++uniqueSelectObjectID;		//For every question, not for answers. Prepare for next question.
}






