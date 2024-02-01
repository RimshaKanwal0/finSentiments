from django.db import IntegrityError
from django.shortcuts import render, redirect, get_object_or_404
from django.urls import reverse
from .models import Dataset
from .forms import DatasetForm
from django.contrib.auth.decorators import login_required
from django.templatetags.static import static
from django.contrib import messages
from django.http import FileResponse


@login_required
def dataset_list(request):
    if request.method == 'POST':
        form = DatasetForm(request.POST, request.FILES)
        if form.is_valid():
            try:
                # attempt to save form or create object
                form.save()
                messages.success(request, 'Entry has been saved successfully.')
            except IntegrityError:
                messages.error(request, 'Something went wrong. Try Again.')
            return redirect(reverse('datasets:dataset_list'))
    else:
        form = DatasetForm()

    datasets = Dataset.objects.all()
    default_thumbnail_url = static('images/default-file-thumbnail.png')
    return render(request, 'datasets/dataset_list.html', {
        'form': form,
        'datasets': datasets,
        'default_thumbnail_url': default_thumbnail_url,
    })


@login_required
def dataset_delete(request, pk):
    dataset = get_object_or_404(Dataset, pk=pk)
    dataset.delete()
    return redirect(reverse('datasets:dataset_list'))


def download_dataset(request, dataset_id):
    dataset = get_object_or_404(Dataset, pk=dataset_id)
    file_path = dataset.file.path
    response = FileResponse(open(file_path, 'rb'), as_attachment=True, filename=dataset.file.name)
    return response
