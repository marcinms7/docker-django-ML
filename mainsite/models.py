from django.db import models

# Create your models here.

class Document(models.Model):
	# description = models.CharField(max_length=200, blank= True)
	document = models.FileField()
	uploaded_at = models.DateTimeField(auto_now_add=True)
	# activated = models.BooleanField(default=False)
	epochs = models.IntegerField(default=10)
	hidden_layers = models.IntegerField(default=2)

	def __str(self):
		return self.id