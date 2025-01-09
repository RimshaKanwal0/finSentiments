from django.shortcuts import render, get_object_or_404, redirect
from .models import AnalysisResult
from .utils import perform_sentiment_analysis, preprocess_data
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.shortcuts import render
from django.http import JsonResponse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import pandas as pd



def perform_analysis(request):
    return render(request, 'perform_analysis')

@login_required
def analysis_input(request):
    # Handle form submission and analysis logic here if needed
    return render(request, 'analysis/analysis_input.html')

from django.shortcuts import render
import pandas as pd
from .preprocess_text import preprocess_text

def perform_analysis(request):
    try:
        # Path to your dataset
        dataset_path = 'E:/django/sentimentproject/finSentiments/dataset/data.csv'
        dataset = pd.read_csv(dataset_path)

        # Check if 'Sentence' column exists
        if 'Sentence' not in dataset.columns:
            return render(request, 'analysis/perform_analysis.html', {
                'message': 'Dataset does not have a "Sentence" column.',
                'success': False
            })

        # Preprocess the 'Sentence' column
        dataset['preprocessed_sentence'] = dataset['Sentence'].apply(preprocess_text)

        # Perform sentiment analysis (dummy example for now)
        # Here, you can update the logic to use the actual sentiment values
        dataset['predicted_sentiment'] = dataset['preprocessed_sentence'].apply(lambda x: 'positive' if 'profit' in x else 'negative')

        # Save the processed dataset
        output_path = 'E:/django/sentimentproject/finSentiments/dataset/processed_data.csv'
        dataset.to_csv(output_path, index=False)

        # Pass data to the template
        return render(request, 'analysis/perform_analysis.html', {
            'dataset': dataset.head(10).to_dict(orient='records'),  # Show first 10 rows
            'message': 'Preprocessing and sentiment analysis completed successfully!',
            'success': True
        })

    except Exception as e:
        # Handle errors gracefully
        return render(request, 'analysis/results.html', {
            'message': f'Error: {str(e)}',
            'success': False
        })

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

# def analysis_result(request, dataset_id):
#     dataset = get_object_or_404(Dataset, id=dataset_id)
#     results = AnalysisResult.objects.filter(dataset=dataset)
#     return render(request, 'analysis/analysis_result.html', {'results': results, 'dataset': dataset})
