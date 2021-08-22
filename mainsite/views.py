from django.shortcuts import render
from django.http import HttpResponse
from mainsite.models import Document
from mainsite.forms import DocumentForm
from django.core.files.storage import FileSystemStorage
from django.contrib import messages
from mainsite.nn_model import extract
import pandas as pd 
 
# Create your views here.

# def index (request):
# 	placeholder = {'test' : 'test'}
# 	documents = Document.objects.all()
# 	return render(request, 'mainsite/csv.html', { 'documents': documents })

AUC = 0 

def model_form_upload(request):
	uploaded = False
	print("Start")
	print(request.method)
	if request.method == 'POST':
		print("HI")
		form = DocumentForm(request.POST, request.FILES )
		if form.is_valid():
			print(request.POST)
			
			csvfile = request.FILES['document']
			if int(request.POST['epochs']) > 1000:
				messages.error(request, 'That exceeds computational power of this program. Please refresh page and define epochs to be lower than 1000')
			if int(request.POST['epochs']) < 0:
				messages.error(request, 'Please choose non-negative number of epochs.')
			if int(request.POST['hidden_layers']) < 2:
				messages.error(request, 'Please choose the number of hidden layers greater than 1.')
			if not csvfile.name.endswith('.csv'):
				messages.error(request, 'THIS IS NOT A CSV FILE! Please refresh page and upload .csv file')
			else:
			# if not form.name.endswith('.csv'):
			# 	messages.error(request, 'THIS IS NOT A CSV FILE')
				form.save()
				form = DocumentForm()
				print("uploaded")
				auc, boolempty = extract(csvfile.name, int(request.POST['epochs']), int(request.POST['hidden_layers']))
				print("extracted")
				if boolempty:
					uploaded = True
				else:
					nodata_message = "Model sucesfully trained with accuracy: " + str(auc) + ". No data to predict included in the file, therefore no uploaded prediction output can be downloaded. If you wish to  download the prediction file, please add data you want to predict, following the instruction above."
					messages.error(request, nodata_message)


	else:
		form = DocumentForm()
		print("GET is here")

	print(form.errors)

	if not uploaded:
		return render(request, 'mainsite/csv.html', {'form' : form})
	else:
		return render(request, 'mainsite/csv_post.html', {'form' : form, 'auc' : auc})
















