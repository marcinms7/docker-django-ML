from django.shortcuts import render


# Create your views here.
def Explanation(request):
	return render(request, 'modelexpl/modelexpl.html')