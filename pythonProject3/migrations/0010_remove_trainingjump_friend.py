# Generated by Django 3.2 on 2023-09-03 02:39

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('pythonProject3', '0009_remove_trainingonefoot_friend'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='trainingjump',
            name='friend',
        ),
    ]
