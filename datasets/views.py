from django.shortcuts import render
from django.http import HttpResponse
from django.contrib.auth.decorators import login_required
from .forms import DatasetForm
# Create your views here.


def example_view(request):
    return HttpResponse("This is an example view.")


@login_required
def upload_dataset(request):
    if request.method == 'POST':
        form = DatasetForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            return redirect('some-view')
    else:
        form = DatasetForm()
    return render(request, 'datasets/upload.html', {'form': form})
