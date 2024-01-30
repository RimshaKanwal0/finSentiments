from django.db import models
from datasets.models import Dataset  # Adjust the import as per your project structure


class AnalysisResult(models.Model):
    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE)
    result = models.JSONField()  # Store results as JSON
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"Analysis for {self.dataset.name}"
