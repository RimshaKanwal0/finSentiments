from django.db import models
from users.models import CustomUser  # Import your custom user model
from base.models import BaseModel  # Import BaseModel


class Dataset(BaseModel):
    user = models.ForeignKey(CustomUser, on_delete=models.CASCADE)
    name = models.CharField(max_length=255)
    file = models.FileField(upload_to='datasets/datasets')

    def __str__(self):
        return self.name
