from django.urls import path
from . import views

app_name = 'datasets'

urlpatterns = [
    path('upload/', views.upload_dataset, name='upload-dataset'),
    path('', views.dataset_list, name='dataset_list'),
    path('delete/<int:pk>/', views.dataset_delete, name='dataset_delete'),

]
