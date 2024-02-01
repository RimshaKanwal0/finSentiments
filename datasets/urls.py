from django.urls import path
from . import views
from .views import download_dataset

app_name = 'datasets'

urlpatterns = [
    path('', views.dataset_list, name='dataset_list'),
    path('delete/<int:pk>/', views.dataset_delete, name='dataset_delete'),
    path('download/<int:dataset_id>/', download_dataset, name='download_dataset'),

]
