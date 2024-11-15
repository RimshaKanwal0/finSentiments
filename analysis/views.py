from django.shortcuts import render, get_object_or_404, redirect
from .models import AnalysisResult
from .utils import perform_sentiment_analysis, preprocess_data
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.shortcuts import render
from django.http import JsonResponse
from .analysis import preprocess_text, train_and_evaluate_blended_classifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import pandas as pd

from .analysis import train_and_evaluate_blended_classifier
import pandas as pd


def perform_analysis(request):
    return render(request, 'perform_analysis')

@login_required
def analysis_input(request):
    # Handle form submission and analysis logic here if needed
    return render(request, 'analysis/analysis_input.html')


def perform_analysis(request):
    if request.method == "POST":
        print("Received POST request for analysis.")  # Log when POST request is received

        # Load the dataset
        try:
            dataset = pd.read_csv('E:/django/sentimentproject/finSentiments/dataset/data.csv')
            print("Dataset loaded successfully.")  # Log dataset loading
        except Exception as e:
            print(f"Error loading dataset: {e}")  # Log error in dataset loading
            return render(request, 'analysis/perform_analysis.html', {'error': f'Error loading dataset: {e}'})

        # Validate dataset columns
        if 'Sentence' not in dataset.columns or 'Sentiment' not in dataset.columns:
            print("Dataset validation failed: Missing required columns.")  # Log validation failure
            return render(request, 'analysis/perform_analysis.html', {'error': "Dataset must contain 'Sentence' and 'Sentiment' columns."})
        print("Dataset validation successful.")  # Log validation success

        # Get form inputs
        selected_preprocessing = request.POST.getlist('preprocessing')
        enable_blending = request.POST.get('blending', False)  # This checks if the blending option is selected
        selected_classifiers = request.POST.getlist('classifier')

        # If blending is enabled, automatically select all classifiers (SVM and SGD)
        if enable_blending:
            selected_classifiers = ['svm', 'sgd']
            print("Blending is enabled. Classifiers automatically set to: SVM, SGD.")  # Log blending logic

        if not selected_classifiers:
            print("No classifiers selected.")  # Log missing classifiers
            return render(request, 'analysis/perform_analysis.html', {'error': "Please select at least one classifier."})

        # Preprocess text based on user selection
        print("Starting text preprocessing...")  # Log preprocessing start
        dataset['Preprocessed_Sentence'] = dataset['Sentence'].apply(
            lambda text: preprocess_text(
                text=text,
                remove_stopwords='remove_stopwords' in selected_preprocessing,
                lemmatization='lemmatization' in selected_preprocessing
            )
        )
        print("Text preprocessing completed.")  # Log preprocessing completion

        # Extract features and labels
        X = dataset['Preprocessed_Sentence']
        y = dataset['Sentiment']
        print("Features and labels extracted.")  # Log feature-label extraction

        # Encode labels
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        print("Labels encoded successfully.")  # Log label encoding

        # Apply TF-IDF vectorization
        tfidf_vectorizer = TfidfVectorizer()
        X_tfidf = tfidf_vectorizer.fit_transform(X)
        print("TF-IDF vectorization completed.")  # Log vectorization

        # Balance the dataset using SMOTE (if blending is enabled)
        if enable_blending:
            smote = SMOTE(random_state=42)
            X_balanced, y_balanced = smote.fit_resample(X_tfidf, y_encoded)
            print("Dataset balanced using SMOTE.")  # Log SMOTE application
        else:
            X_balanced, y_balanced = X_tfidf, y_encoded
            print("Dataset balancing skipped.")  # Log skipping of SMOTE

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)
        print("Data split into training and testing sets.")  # Log data splitting

        # Train and evaluate using selected classifiers
        try:
            print("Starting classifier training and evaluation...")  # Log start of training
            results = train_and_evaluate_blended_classifier(
                classifier_combo=selected_classifiers,
                additional_info={'Enable Blending': enable_blending},
                num_classifiers=len(selected_classifiers)
            )
            print("Classifier training and evaluation completed.")  # Log training completion
        except Exception as e:
            print(f"Error during analysis: {e}")  # Log error in analysis
            return render(request, 'analysis/perform_analysis.html', {'error': f"Error during analysis: {e}"})

        # Render success message with results
        print("Analysis completed successfully.")  # Log successful completion
        return render(
            request,
            'analysis/perform_analysis.html',
            {
                'success': "Analysis completed successfully!",
                'results': results,
                'total_classifiers': len(selected_classifiers)
            }
        )

    # Render the default form page
    print("Rendering analysis form page.")  # Log rendering of form page
    return render(request, 'analysis/perform_analysis.html')



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
