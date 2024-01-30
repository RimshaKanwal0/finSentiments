from django.db import models
from django.utils import timezone
from users.models import CustomUser


class BaseModel(models.Model):
    created_at = models.DateTimeField(auto_now_add=True)
    created_by = models.ForeignKey(CustomUser, related_name="%(class)s_created", on_delete=models.CASCADE)
    updated_at = models.DateTimeField(auto_now=True)
    updated_by = models.ForeignKey(CustomUser, related_name="%(class)s_updated", on_delete=models.CASCADE)

    class Meta:
        abstract = True
