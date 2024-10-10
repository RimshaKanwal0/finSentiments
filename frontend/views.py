# E:\django\sentimentproject\finSentiments\frontend\views.py
from django.shortcuts import render

# Create your views here.
from django.shortcuts import render
from django.http import HttpResponse

def submit_contact(request):
    if request.method == 'POST':
        # Handle form submission logic here (e.g., save form data)
        return HttpResponse("Contact form submitted!")
    else:
        return render(request, 'contact.html')  # Assuming there's a 'contact.html' template

def contact(request):
    return render(request, 'contact.html')  # Assuming there's a 'contact.html' template

def home(request):
    return render(request, 'frontend/home.html')  # Correct path for the template
