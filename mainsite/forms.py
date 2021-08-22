from django import forms
from mainsite.models import Document

class DocumentForm(forms.ModelForm):
	class Meta:
		model = Document
		fields = (
			# 'description',
			'epochs',
			'hidden_layers',
			 'document',
			 )