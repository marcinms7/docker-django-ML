from django.shortcuts import render
from django.http import HttpResponse
from django.urls import path
from mainsite import views
# Create your views here.

app_name = 'mainsite'

urlpatterns = [
path('', views.model_form_upload, name='model_form_upload'),
# path('', views.index, name='index')
]