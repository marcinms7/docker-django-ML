

var myVar;
myVar = setTimeout(showPage, 3000);


function showPage() {
  document.getElementById("loader").style.display = "none";
  document.getElementById("myDiv").style.display = "block";

  
	var headOne = document.querySelector('#one')

	headOne.addEventListener('mouseover', function(){
	  headOne.textContent = 'MAKE THIS ANIMATED!'
	}
	)

	headOne.addEventListener('mouseout', function(){
	  headOne.textContent = 'LOADING MODEL '
	}
	)



}






