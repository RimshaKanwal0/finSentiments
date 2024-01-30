from django.urls import path
from . import views

app_name = 'analysis'

urlpatterns = [

    path('analysis/input/', views.analysis_input, name='analysis_input'),
    path('perform-analysis/', views.analysis_input, name='perform-analysis'),
    path('analysis/progress/', views.analysis_progress, name='analysis_progress'),
    # path('perform-analysis/<int:dataset_id>/', views.perform_analysis, name='perform-analysis'),
    path('results/<int:dataset_id>/', views.analysis_result, name='analysis_result'),

]
