from django.db import models
from django.core.validators import FileExtensionValidator
from django.db.models.signals import post_save
from django.dispatch import receiver
from PIL import Image, ImageDraw, ImageFont
import os
from io import BytesIO
from django.core.files import File
from base.models import BaseModel  # Import BaseModel


class Dataset(BaseModel):
    name = models.CharField(max_length=255)
    file = models.FileField(upload_to='datasets/',
                            validators=[FileExtensionValidator(allowed_extensions=['csv', 'xlsx'])])
    thumbnail = models.ImageField(upload_to='thumbnails/', blank=True, null=True)

    def __str__(self):
        return self.name

    class Meta:
        constraints = [
            models.UniqueConstraint(fields=['name', 'file'], name='unique_dataset')
        ]

    def save(self, *args, **kwargs):
        super().save(*args, **kwargs)
        if not self.thumbnail:
            self.generate_thumbnail()

    def generate_thumbnail(self):
        # Create a thumbnail image (this is a placeholder - update as needed)
        img = Image.new('RGB', (100, 50), color=(73, 109, 137))
        d = ImageDraw.Draw(img)
        d.text((10, 10), "CSV", fill=(255, 255, 0))

        # Save thumbnail to in-memory file as StringIO
        thumb_io = BytesIO()
        img.save(thumb_io, 'PNG', quality=85)

        thumbnail_name = os.path.basename(self.file.name)
        thumbnail_extension = '.png'
        thumbnail_filename = thumbnail_name + thumbnail_extension

        # Set save=False, otherwise it will run in an infinite loop
        self.thumbnail.save(thumbnail_filename, File(thumb_io), save=False)
        self.save()


# If you prefer to use a signal instead of overriding save method
@receiver(post_save, sender=Dataset)
def create_thumbnail(sender, instance, created, **kwargs):
    if created:
        instance.generate_thumbnail()
