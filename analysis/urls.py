from django.urls import path
from . import views

urlpatterns = [

    path('perform-analysis/<int:dataset_id>/', views.perform_analysis, name='perform-analysis'),
    path('results/<int:dataset_id>/', views.analysis_result, name='analysis_result'),

]
