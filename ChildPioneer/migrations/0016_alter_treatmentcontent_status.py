# Generated by Django 3.2 on 2023-09-06 16:48

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('ChildPioneer', '0015_customuser_background_images'),
    ]

    operations = [
        migrations.AlterField(
            model_name='treatmentcontent',
            name='status',
            field=models.BooleanField(default=False),
        ),
    ]
