from django.db import models
from django.core.validators import FileExtensionValidator
from django.db.models.signals import post_save
from django.dispatch import receiver
from PIL import Image, ImageDraw, ImageFont
import os
from django.utils.timezone import now
from io import BytesIO
from django.core.files import File
from base.models import BaseModel  # Import BaseModel
from django.conf import settings
from django.core.files.base import ContentFile


def dataset_directory_path(instance, filename):
    # Extract the file extension and prepare the new filename
    extension = os.path.splitext(filename)[1]
    new_filename = f"{instance.name}_{now().strftime('%Y%m%d%H%M%S')}{extension}"
    return os.path.join('datasets', new_filename)


class Dataset(BaseModel):
    name = models.CharField(max_length=255, unique=True)
    file = models.FileField(upload_to=dataset_directory_path,
                            validators=[FileExtensionValidator(allowed_extensions=['csv', 'xlsx'])])
    thumbnail = models.ImageField(upload_to='thumbnails/', blank=True, null=True)
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='created_datasets'
    )

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
        # Assuming the file is a CSV and is not too large
        if self.file.name.endswith('.csv'):
            with self.file.open('r') as file:
                content = file.read(1024)  # Read the first 1KB of the file

            # Create a new image with PIL
            img = Image.new('RGB', (100, 50), color=(255, 255, 255))
            draw = ImageDraw.Draw(img)
            draw.text((10, 10), content, fill=(0, 0, 0))

            # Save the image to a BytesIO object
            thumb_io = BytesIO()
            img.save(thumb_io, format='JPEG')

            # Create a Django File from the BytesIO object
            thumbnail = ContentFile(thumb_io.getvalue(), name='thumbnail.jpg')
            self.thumbnail.save(name='thumbnail.jpg', content=thumbnail, save=False)

        super().save()


# If you prefer to use a signal instead of overriding save method
@receiver(post_save, sender=Dataset)
def create_thumbnail(sender, instance, created, **kwargs):
    if created:
        instance.generate_thumbnail()
