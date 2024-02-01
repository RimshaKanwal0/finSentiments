from django.urls import path
from . import views

app_name = 'analysis'

urlpatterns = [

    path('analysis/input/', views.analysis_input, name='analysis_input'),
    path('analysis/progress/', views.analysis_progress, name='analysis_progress'),

    path('perform-analysis/', views.perform_analysis, name='perform-analysis'),
    path('perform-analysis/<int:training_dataset_id>/<int:testing_dataset_id>/', views.perform_analysis,
         name='perform_analysis_with_dataset'),
    # path('perform-analysis/<int:dataset_id>/', views.perform_analysis, name='perform-analysis'),

    path('results/<int:dataset_id>/', views.analysis_result, name='analysis_result'),

]
