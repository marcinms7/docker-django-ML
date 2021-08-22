from django.shortcuts import render
from django.urls import path
from contact import views
# Create your views here.

urlpatterns = [
    path('', views.Contact, name = 'contact')]