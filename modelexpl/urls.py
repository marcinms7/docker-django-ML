from django.shortcuts import render
from django.urls import path
from modelexpl import views
# Create your views here.

urlpatterns = [
path('', views.Explanation, name='modelexpl')]