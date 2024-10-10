from django.shortcuts import render, get_object_or_404, redirect
from .models import AnalysisResult, Dataset
from .utils import perform_sentiment_analysis, preprocess_data
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from .preprocess import preprocess_text
import pandas as pd


def perform_analysis(request):
    return render(request, 'perform_analysis')

@login_required
def analysis_input(request):
    # Handle form submission and analysis logic here if needed
    return render(request, 'analysis/analysis_input.html')

@login_required
def analysis_progress(request):
    return render(request, 'analysis/analysis_progress.html')

#main logic

def perform_analysis(request):
    if request.method == 'POST':
        remove_stopwords = 'remove_stopwords' in request.POST.getlist('preprocessing')
        lemmatization = 'lemmatization' in request.POST.getlist('preprocessing')

        # Load your dataset (adjust the path as needed)
        df = pd.read_csv('E:/django/sentimentproject/finSentiments/dataset/data.csv')

        # Apply preprocessing
        df['Preprocessed_Sentence'] = df['Sentence'].apply(
            lambda x: preprocess_text(x, remove_stopwords=remove_stopwords, lemmatization=lemmatization)
        )

        # Print or process the preprocessed data (e.g., save or pass it to the template)
        print(df['Preprocessed_Sentence'].head())

        # Optionally save the results back to a file
        df.to_csv('E:/django/sentimentproject/finSentiments/dataset/preprocessed_data.csv', index=False)

        # Confirmation message or redirect
        return render(request, 'analysis/perform_analysis.html', {'success': 'Preprocessing completed successfully!'})

    return render(request, 'analysis/perform_analysis.html')

#
# def perform_analysis(request, training_dataset_id=None, testing_dataset_id=None):
#     selected_dataset = None
#     training_dataset = None
#     datasets = Dataset.objects.all()
#     if request.method == 'POST':
#         training_dataset_id = request.POST.get('training_dataset')
#         testing_dataset_id = request.POST.get('testing_dataset')
#         # Logic to handle analysis using the selected datasets
#         # You might want to pass these datasets to your analysis function
#
#     if training_dataset_id:
#         training_dataset = get_object_or_404(Dataset, pk=training_dataset_id)
#         testing_dataset = get_object_or_404(Dataset, pk=testing_dataset_id)
#     return render(request, 'analysis/perform_analysis.html',
#                   {'datasets': datasets, 'training_dataset': training_dataset,
#                    'testing_dataset': testing_dataset})
#
#
# @login_required
# def perform_analysis(request):
#     if request.method == 'POST':
#         dataset_id = request.POST.get('dataset')
#         dataset = get_object_or_404(Dataset, id=dataset_id)
#         preprocessing_techniques = request.POST.getlist('preprocessing')
#         blending = 'blending' in request.POST
#         classifiers = request.POST.getlist('blend_classifier') if blending else request.POST.get('classifier')
#
#         # Initialize preprocessed_data
#         preprocessed_data = None
#
#         # If dataset and preprocessing techniques are defined, preprocess data
#         if dataset and preprocessing_techniques:
#             preprocessed_data = preprocess_data(dataset, preprocessing_techniques)
#
#         # Proceed with analysis if preprocessed_data is available
#         if preprocessed_data is not None:
#             analysis_result = perform_sentiment_analysis(preprocessed_data, classifiers, blending)
#             AnalysisResult.objects.create(dataset=dataset, result=analysis_result)
#             return redirect('analysis_result', dataset_id=dataset.id)
#
#     datasets = Dataset.objects.all()  # Assuming user has permission to view all datasets
#     return render(request, 'analysis/perform_analysis.html', {'datasets': datasets})
#

def analysis_result(request, dataset_id):
    dataset = get_object_or_404(Dataset, id=dataset_id)
    results = AnalysisResult.objects.filter(dataset=dataset)
    return render(request, 'analysis/analysis_result.html', {'results': results, 'dataset': dataset})
