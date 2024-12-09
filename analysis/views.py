from django.shortcuts import render, get_object_or_404, redirect
from .models import AnalysisResult
from .utils import perform_sentiment_analysis, preprocess_data
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.shortcuts import render
from django.http import JsonResponse
# from .analysis import preprocess_text, train_and_evaluate_blended_classifier
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


# View for rendering the analysis page
def analysis_view(request):
    # You can pass additional context here if needed for rendering templates
    return render(request, 'analysis/perform_analysis.html')

def perform_analysis(request):
    if request.method == 'POST':
        preprocessing = request.POST.get('preprocessing')
    return render(request, 'analysis/perform_analysis.html')

# simple post
# def perform_analysis(request):
#     if request.method == "POST":
#         preprocessing_tec = request.POST.getlist("preprocessing")
#         blending = request.POST.get("blending", None) == "on"
#         classifiers = request.POST.getlist("classifier")
#         # response
#         data = {
#             "preprocessing_techniques": preprocessing_tec,
#             "blending": blending,
#             "classifiers": classifiers,
#         }
#         return JsonResponse(data)
#     return render(request, "analysis/perform_analysis.html")

from django.shortcuts import render
from django.http import JsonResponse
import pandas as pd
from .analysis import preprocess_text  # Import the preprocess_text function
import os

def perform_analysis(request):
    if request.method == 'POST':
        try:
            # Get preprocessing options from the form
            preprocessing_options = request.POST.getlist('preprocessing')

            # Load dataset (you can modify this to dynamically accept uploaded files)
            dataset_path = 'E:/django/sentimentproject/finSentiments/dataset/data.csv'
            if not os.path.exists(dataset_path):
                return render(request, 'analysis/perform_analysis.html', {
                    'error': 'Dataset not found at the specified location.'
                })

            df = pd.read_csv(dataset_path)

            # Apply preprocessing based on selected options
            remove_stopwords = 'remove_stopwords' in preprocessing_options
            lemmatization = 'lemmatization' in preprocessing_options

            # Add a new column with preprocessed sentences
            df['Preprocessed_Sentence'] = df['Sentence'].apply(
                lambda x: preprocess_text(x,
                                          remove_stopwords=remove_stopwords,
                                          lemmatization=lemmatization)
            )

            # Save the preprocessed results back to a CSV file (optional)
            output_path = 'E:/django/sentimentproject/finSentiments/dataset/preprocessed_data.csv'
            df.to_csv(output_path, index=False)

            # Success message
            return render(request, 'analysis/perform_analysis.html', {
                'success': 'Preprocessing completed successfully! Results saved to preprocessed_data.csv.'
            })

        except Exception as e:
            # Handle errors and render them
            return render(request, 'analysis/perform_analysis.html', {
                'error': f'An error occurred during preprocessing: {str(e)}'
            })

    # Render the form on GET request
    return render(request, 'analysis/perform_analysis.html')




# Rough...
# logic of perform-analysis fun take from analysis.py
# from django.shortcuts import render
# from django.http import JsonResponse
# from .analysis import preprocess_and_split_data, train_and_evaluate_blended_classifier
# from itertools import combinations
#
# # Define available classifiers
# from sklearn.linear_model import SGDClassifier
# from sklearn.svm import SVC
#
# classifiers = {
#     'SVM': SVC(),
#     'SGD': SGDClassifier(),
# }
#
# # Perform Analysis View
# def perform_analysis(request):
#     success = None
#     error = None
#     results = None
#
#     if request.method == 'POST':
#         try:
#             print("Starting the analysis process...")
#
#             # Load dataset
#             print("Loading dataset...")
#             import pandas as pd
#             df = pd.read_csv('E:/django/sentimentproject/finSentiments/dataset/data.csv')
#             print("Dataset loaded successfully.")
#
#             # Preprocess the data
#             print("Starting data preprocessing...")
#             preprocessing_flags = {
#                 'remove_stopwords': 'remove_stopwords' in request.POST,
#                 'lemmatization': 'lemmatization' in request.POST,
#             }
#             X_train, X_test, y_train, y_test, tfidf_vectorizer, label_encoder = preprocess_and_split_data(
#                 df, preprocessing_flags, smote_flag=True, test_size=0.2, rand_state=42
#             )
#             print("Data preprocessing completed.")
#
#             # Handle blending and classifier selection
#             print("Processing classifier selection...")
#             use_blending = 'blending' in request.POST
#             selected_classifiers = request.POST.getlist('classifier')
#
#             if not selected_classifiers and not use_blending:
#                 raise ValueError("Please select at least one classifier or enable blending.")
#
#             print(f"Blending enabled: {use_blending}")
#             print(f"Selected classifiers: {selected_classifiers}")
#
#             # Prepare classifier combinations for blending
#             classifier_combinations = (
#                 combinations(selected_classifiers, 2) if use_blending else [(clf,) for clf in selected_classifiers]
#             )
#
#             # Train and evaluate
#             print("Starting training and evaluation...")
#             results = []
#             for combo in classifier_combinations:
#                 print(f"Training with classifier(s): {combo}")
#                 result = train_and_evaluate_blended_classifier(
#                     classifier_combo=combo,
#                     classifiers=classifiers,
#                     X_train=X_train,
#                     X_test=X_test,
#                     y_train=y_train,
#                     y_test=y_test,
#                 )
#                 print(f"Evaluation completed for classifier(s): {combo}")
#                 results.append(result)
#
#             print("All training and evaluation processes completed.")
#             success = "Analysis completed successfully!"
#         except Exception as e:
#             error = f"An error occurred: {str(e)}"
#             print(f"Error: {error}")
#
#     # Render results in the template
#     return render(request, 'analysis/results.html', {
#         'success': success,
#         'error': error,
#         'results': results,
#         'total_classifiers': len(results) if results else 0,
#     })


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

#    def analysis_result(request, dataset_id):
#     dataset = get_object_or_404(Dataset, id=dataset_id)
#     results = AnalysisResult.objects.filter(dataset=dataset)
#     return render(request, 'analysis/analysis_result.html', {'results': results, 'dataset': dataset})
