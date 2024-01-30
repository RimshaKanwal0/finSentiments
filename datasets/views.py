from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from .forms import DatasetForm


@login_required
def upload_dataset(request):
    if request.method == 'POST':
        form = DatasetForm(request.POST, request.FILES)
        if form.is_valid():
            dataset = form.save(commit=False)
            dataset.created_by = request.user
            dataset.updated_by = request.user
            dataset.save()
            return redirect('some-view')
    else:
        form = DatasetForm()
    return render(request, 'datasets/upload.html', {'form': form})
