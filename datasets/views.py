from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from .forms import DatasetForm
from .models import Dataset


@login_required
def upload_dataset(request):
    if request.method == 'POST':
        form = DatasetForm(request.POST, request.FILES)
        if form.is_valid():
            dataset = form.save(commit=False)
            dataset.created_by = request.user
            dataset.updated_by = request.user
            dataset.save()
            return redirect('datasets:dataset_list')
    else:
        form = DatasetForm()
    return render(request, 'datasets/upload.html', {'form': form})


def dataset_list(request):
    datasets = Dataset.objects.all()
    return render(request, 'datasets/dataset_list.html', {'datasets': datasets})


def dataset_delete(request, pk):
    Dataset.objects.get(pk=pk).delete()
    return redirect('datasets:dataset_list')