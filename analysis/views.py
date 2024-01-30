from django.shortcuts import render, get_object_or_404, redirect
from .models import AnalysisResult, Dataset
from .utils import perform_sentiment_analysis, preprocess_data
from django.contrib.auth.decorators import login_required


@login_required
def perform_analysis(request):
    if request.method == 'POST':
        dataset_id = request.POST.get('dataset')
        dataset = get_object_or_404(Dataset, id=dataset_id)
        preprocessing_techniques = request.POST.getlist('preprocessing')
        blending = 'blending' in request.POST
        classifiers = request.POST.getlist('blend_classifier') if blending else request.POST.get('classifier')

        # Initialize preprocessed_data
        preprocessed_data = None

        # If dataset and preprocessing techniques are defined, preprocess data
        if dataset and preprocessing_techniques:
            preprocessed_data = preprocess_data(dataset, preprocessing_techniques)

        # Proceed with analysis if preprocessed_data is available
        if preprocessed_data is not None:
            analysis_result = perform_sentiment_analysis(preprocessed_data, classifiers, blending)
            AnalysisResult.objects.create(dataset=dataset, result=analysis_result)
            return redirect('analysis_result', dataset_id=dataset.id)

    datasets = Dataset.objects.all()  # Assuming user has permission to view all datasets
    return render(request, 'analysis/perform_analysis.html', {'datasets': datasets})
