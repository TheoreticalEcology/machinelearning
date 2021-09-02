/*
*	This script is for generating multiple choice questions.
*	Version 0.992
*/


//Try to not include this script several times in one file. That means, use <script src=".../multipleChoice.js"></script> only one time (if possible).
//The function 'makeMultipleChoiceForm' may be used as often as you like.


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
*
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
*		Indicator of correctness is named like the respective object, followed by '_correctness'. For example 'checkbox_2_correctness'.
*
*		Answer labels are named like explanations but with '_answerField' instead of any explanation. For example 'radio_0$1_answerField'
*	
*/


////////////////////////////////////////////////////////////
////////////////////	Global variables
////////////////////////////////////////////////////////////

if(multipleChoiceIncluded === undefined){	//Prevent from errors, if the script is included multiple times.
	var multipleChoiceIncluded = true;	//Define variable 'multipleChoiceIncluded' if already included.
	var uniqueSelectObjectID = 0;		//Global(!) counter for inserted multiple choice forms.
	var correctAnswers = {};			//Global variable for storing (indices of) correct answers.
	var textAnswers = {};				//Global variable for storing all supplied text answers.
}


//The following variables apply to the WHOLE site (not only for 1 question). Change them carefully.

var dyeAnswers = true;					//true/false. Indicates whether wrong answers (rather the respective labels) should get red and right ones green.
var radioDyeAll = false;				//true/false. Set to true, if you want to show correctness (dye labels) of ALL (not just the selected) answer.
										//Only usefull, if 'dyeAnswers' is true.
var disableButtonOnClick = true;		//true/false. Set to true, if you want to enable only 1 button click (disabled after).

////////////////////////////////////////////////////////////
////////////////////	End global variables
////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////
////////////////////	Helper functions
////////////////////////////////////////////////////////////

/*!
*	\brief	Makes entered invisible text visible, if not empty.
*			This function is used very often and thus 'outsourced'.
*	
*	@param	uniqueIdentifier[in]	Unique object identifier (literally; by name).
*/
function makeVisible(uniqueIdentifier){
	innerElement = document.getElementById(uniqueIdentifier);
	if(innerElement.innerHTML != ''){ innerElement.style.display = 'block'; }	//Do not display empty elements.
}


/*!
*	\brief	If desired, wrong answers (rather the respective labels) get red and right ones green.
*	
*	@param	uniqueIdentifier[in]	Unique object identifier of the answer (literally; by name).
*	@param	color[in]				Color to dye the label. Should be 'red' or 'green' (or 'black' for resetting) but other colors are ok as well.
*/
function dyeAnswerLabel(uniqueIdentifier, color){
	if(dyeAnswers){ document.getElementById(uniqueIdentifier + '_answerField').style.color = color; }
}

////////////////////////////////////////////////////////////
////////////////////	End helper functions
////////////////////////////////////////////////////////////


/*!
*	\brief	Is called when a submit button was clicked.
*			Evaluates the given answers and makes the corresponding explanation and comment fields visible.
*			Makes use of the global variables 'correctAnswers' and 'textAnswers'.
*	
*	@param	objectID[in]	Global question ID with kind prefix (kind_id).
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
				if(correctAnswerIndices.includes(i)){ userWasRight = true; }
				makeVisible(objectID + '$' + i + '_explanationIfSelected');
			}else{ makeVisible(objectID + '$' + i + '_explanationIfNotSelected'); }
			makeVisible(objectID + '$' + i + '_explanationGeneral');
		}
	}else if(kind == 'radio'){
		let i = 0;
		while(element = document.getElementById(objectID + '$' + i)){
			if(element.checked){
				if(correctAnswerIndices.includes(i)){
					userWasRight = true;	//1 correct answer is enough.
					dyeAnswerLabel(objectID + '$' + i, 'green');
				}else{ dyeAnswerLabel(objectID + '$' + i, 'red'); }
				//Maybe explanations should be displayed, so do not break the loop here.
				makeVisible(objectID + '$' + i + '_explanationIfSelected');
			}else{
				if(radioDyeAll){	//If you want to show correctness (dye labels) of ALL (not just the selected) answer.
					if(!correctAnswerIndices.includes(i)){ dyeAnswerLabel(objectID + '$' + i, 'red'); }
					else{ dyeAnswerLabel(objectID + '$' + i, 'green'); }
				}else{ dyeAnswerLabel(objectID + '$' + i, 'black'); }
				makeVisible(objectID + '$' + i + '_explanationIfNotSelected');
			}
			makeVisible(objectID + '$' + i + '_explanationGeneral');
			++i;
		}
	}else if(kind == 'checkbox'){
		userWasRight = true;
		let i = 0;
		while(element = document.getElementById(objectID + '$' + i)){
			if(element.checked){
				if(!correctAnswerIndices.includes(i)){
					userWasRight = false;	//All anserws must be correct.
					dyeAnswerLabel(objectID + '$' + i, 'red');
				}else{ dyeAnswerLabel(objectID + '$' + i, 'green'); }
				makeVisible(objectID + '$' + i + '_explanationIfSelected');
			}else{
				if(correctAnswerIndices.includes(i)){
					userWasRight = false;	//All anserws must be correct.
					dyeAnswerLabel(objectID + '$' + i, 'red');
				}else{ dyeAnswerLabel(objectID + '$' + i, 'green'); }
				makeVisible(objectID + '$' + i + '_explanationIfNotSelected');
			}
			makeVisible(objectID + '$' + i + '_explanationGeneral');
			++i;
		}
	}
	correctLabel = document.getElementById(objectID + '_correctness');
	correctLabel.innerText = 'Your answer was ' + (userWasRight ? 'correct.' : 'wrong.');
	correctLabel.style.color = (userWasRight ? 'green' : 'red');
	
	makeVisible(kind + '_' + uniqueID + '_comment');
}


/*!
*	\brief	Generates a new (multiple choice) question form.
*			Just use this function in a script tag.
*
*			As JavaScript does not support named parameters, you have to unfortunately use positional arguments.
*
*			Using this script with markdown might lead to some minor but strange formatting issues.
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
	if(kind != 'radio' && kind != 'checkbox' && kind != 'text'){ return; }
	//Omit further type checking as I assume it would not be neccessary.
	
	
	//Locate position and insert new div as container for question and answers.
	var arrScripts = document.getElementsByTagName('script');	//Holds all script tags until now.
	var currScript = arrScripts[arrScripts.length - 1];			//Holds the current script tag.
	var thisDiv = document.createElement('div');	//Unnamed.
	currScript.parentNode.appendChild(thisDiv);	
	
	
	var questionHeading = document.createElement('H4');
	questionHeading.insertAdjacentHTML('afterbegin', question);
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
			explanation.insertAdjacentHTML('afterbegin', element['explanationIfSelected']);
			explanation.id = identifier + localElementCounter + '_explanationIfSelected';
			explanation.style.display = 'none';		//Invisible.
			//explanation.style.display = 'block';	//Visible.
			thisDiv.appendChild(explanation);
			
			var explanation = document.createElement('p');
			explanation.insertAdjacentHTML('afterbegin', element['explanationIfNotSelected']);
			explanation.id = identifier + localElementCounter + '_explanationIfNotSelected';
			explanation.style.display = 'none';		//Invisible.
			//explanation.style.display = 'block';	//Visible.
			thisDiv.appendChild(explanation);
			
			var explanation = document.createElement('p');
			explanation.insertAdjacentHTML('afterbegin', element['explanationGeneral']);
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
			label.id = identifier + '_answerField';
			label.htmlFor = identifier;
			label.insertAdjacentHTML('afterbegin', '  ' + element['answer']);
			
			var newline = document.createElement('br');
			
			thisDiv.appendChild(selectbox);
			thisDiv.appendChild(label);
			thisDiv.appendChild(newline);
			
			
			var explanation = document.createElement('p');
			explanation.insertAdjacentHTML('afterbegin', element['explanationIfSelected']);
			explanation.id = identifier + '_explanationIfSelected';
			explanation.style.display = 'none';		//Invisible.
			//explanation.style.display = 'block';	//Visible.
			thisDiv.appendChild(explanation);
			
			var explanation = document.createElement('p');
			explanation.insertAdjacentHTML('afterbegin', element['explanationIfNotSelected']);
			explanation.id = identifier + '_explanationIfNotSelected';
			explanation.style.display = 'none';		//Invisible.
			//explanation.style.display = 'block';	//Visible.
			thisDiv.appendChild(explanation);
			
			var explanation = document.createElement('p');
			explanation.insertAdjacentHTML('afterbegin', element['explanationGeneral']);
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
	explanation.insertAdjacentHTML('afterbegin', comment);
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
		if(disableButtonOnClick){ this.disabled = true; }
		evaluateQuestion(this.id);
	});
	thisDiv.appendChild(submitButton);
	
	
	var explanation = document.createElement('p');
	var text = document.createTextNode('\xa0');		//Change with document.getElementById(...).innerText = '...'; \xa0 is non breaking space.
	explanation.appendChild(text);
	explanation.id = kind + '_' + uniqueSelectObjectID + '_correctness';
	explanation.style.marginTop = '25px';
	thisDiv.appendChild(explanation);
	
	
	var newline = document.createElement('br');
	thisDiv.appendChild(newline);
	
	correctAnswers[uniqueSelectObjectID] = answerArray;
	
	++uniqueSelectObjectID;		//For every question, not for answers. Prepare for next question.
}






