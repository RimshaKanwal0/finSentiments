from django.db import models
from users.models import CustomUser  # Import your custom user model


class Dataset(models.Model):
    user = models.ForeignKey(CustomUser, on_delete=models.CASCADE)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    name = models.CharField(max_length=255)
    file = models.FileField(upload_to='datasets/')

    def __str__(self):
        return self.name
