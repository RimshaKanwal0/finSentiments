from django.shortcuts import render, redirect, get_object_or_404
from django.urls import reverse
from .models import Dataset
from .forms import DatasetForm


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
