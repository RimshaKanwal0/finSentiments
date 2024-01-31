from django.shortcuts import render, redirect, get_object_or_404
from django.urls import reverse
from .models import Dataset
from .forms import DatasetForm
from django.contrib.auth.decorators import login_required


@login_required
def upload_dataset(request):
    if request.method == 'POST':
        form = DatasetForm(request.POST, request.FILES)
        if form.is_valid():
            dataset = form.save(commit=False)
            dataset.created_by = request.user
            dataset.save()
            return redirect('datasets:dataset_list')
    else:
        form = DatasetForm()
    return render(request, 'datasets/upload_dataset.html', {'form': form})


def dataset_list(request):
    if request.method == 'POST':
        form = DatasetForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            return redirect(reverse('datasets:dataset_list'))
    else:
        form = DatasetForm()

    datasets = Dataset.objects.all()
    return render(request, 'datasets/dataset_list.html', {
        'form': form,
        'datasets': datasets
    })


def dataset_delete(request, pk):
    dataset = get_object_or_404(Dataset, pk=pk)
    dataset.delete()
    return redirect(reverse('datasets:dataset_list'))
