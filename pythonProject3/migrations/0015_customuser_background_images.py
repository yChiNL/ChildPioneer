# Generated by Django 3.2 on 2023-09-06 06:08

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('pythonProject3', '0014_alter_backgroundimage_image_path'),
    ]

    operations = [
        migrations.AddField(
            model_name='customuser',
            name='background_images',
            field=models.ManyToManyField(blank=True, to='pythonProject3.BackgroundImage'),
        ),
    ]
