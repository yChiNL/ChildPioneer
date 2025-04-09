from django.contrib.auth import get_user_model
from django.contrib.auth.base_user import BaseUserManager
from django.db import models
import json
from django.contrib import admin
from django.core.validators import MinLengthValidator
from django.dispatch import receiver
from django.utils import timezone
from django.contrib.auth.models import AbstractUser, Group, Permission
from django.db.models.signals import post_save

class BackgroundImage(models.Model):
    name = models.CharField(max_length=100)
    image_path = models.ImageField(upload_to='background_images')
    price = models.IntegerField()

    def __str__(self):
        return self.name


class CustomUser(AbstractUser):
    USER_TYPES = (
        ('user', 'User'),
        ('therapist', 'Therapist'),
    )

    child_name = models.CharField(max_length=100, default='')
    therapist_name = models.CharField(max_length=100, default='')
    email = models.EmailField(max_length=100)
    user_type = models.CharField(max_length=20, choices=USER_TYPES, default='')
    friends = models.ManyToManyField('self', blank=True, related_name='user_friends')
    therapists = models.ManyToManyField('self', blank=True, related_name='therapist_friends')
    assigned_move_count = models.PositiveIntegerField(default=0)
    parent_name = models.CharField(max_length=50, default='')  # 家長姓名
    parent_phonenumber = models.CharField(max_length=10, default='')
    parent_line = models.CharField(max_length=120, default='')  # 家長line ID
    therapist_phonenumber = models.CharField(max_length=10, default='')
    therapist_line = models.CharField(max_length=120, default='')  # 家長line ID
    coins = models.PositiveIntegerField(default=0)  # 添加 coins 字段
    background_images = models.ManyToManyField(BackgroundImage, blank=True)  # 使用 ManyToManyField 来关联用户和背景图片
    # 添加其他必要的字段
    def __str__(self):
        return self.username
@admin.register(CustomUser)
class UserAdmin(admin.ModelAdmin):
    list_display = [field.name for field in CustomUser._meta.fields]#python幫填寫欄位

class User_BackgroundImage(models.Model):
    user = models.ForeignKey(CustomUser, on_delete=models.CASCADE)
    image_id = models.IntegerField(default=0)
    name = models.CharField(max_length=100)
    image_path = models.ImageField(upload_to='background_images')
    useornot = models.BooleanField(default=False)
    def __str__(self):
        return self.name
@admin.register(User_BackgroundImage)
class User_BackgroundImageAdmin(admin.ModelAdmin):
    list_display = [field.name for field in User_BackgroundImage._meta.fields]#python幫填寫欄位

class TrainingDance(models.Model):
    user = models.ForeignKey(CustomUser, on_delete=models.CASCADE)
    move_count = models.IntegerField()
    t_time = models.IntegerField(default=0)
    difficulty = models.CharField(max_length=10, default='none')
    grade = models.CharField(max_length=1, default='none')
    run_time = models.DateTimeField(default=timezone.now)
@admin.register(TrainingDance)
class TrainingDanceAdmin(admin.ModelAdmin):
    list_display = [field.name for field in TrainingDance._meta.fields]#python幫填寫欄位

class TrainingOneFoot(models.Model):
    user = models.ForeignKey(CustomUser, on_delete=models.CASCADE)
    move_count = models.IntegerField()
    fail_count = models.IntegerField(default=0)
    t_time = models.IntegerField(null=True, blank=True, default=0)
    difficulty = models.CharField(max_length=10, default='none')
    grade = models.CharField(max_length=1, default='none')
    run_time = models.DateTimeField(default=timezone.now)
@admin.register(TrainingOneFoot)
class TrainingOneFootAdmin(admin.ModelAdmin):
        list_display = [field.name for field in TrainingOneFoot._meta.fields]  # python幫填寫欄位

class TrainingJump(models.Model):
    user = models.ForeignKey(CustomUser, on_delete=models.CASCADE)
    move_count = models.IntegerField()
    t_time = models.IntegerField(default=0)
    difficulty = models.CharField(max_length=10, default='none')
    run_time = models.DateTimeField(default=timezone.now)
    grade = models.CharField(max_length=1, default='none')
@admin.register(TrainingJump)
class TrainingJumpAdmin(admin.ModelAdmin):
    list_display = [field.name for field in TrainingJump._meta.fields]#python幫填寫欄位

class TrainingWalkStraight(models.Model):
    user = models.ForeignKey(CustomUser, on_delete=models.CASCADE,null=True)
    move_count = models.IntegerField()
    failure_count = models.IntegerField()
    t_time = models.IntegerField(null=True, blank=True, default=0)
    difficulty = models.CharField(max_length=10, default='none')
    grade = models.CharField(max_length=1, default='none')
    run_time = models.DateTimeField(default=timezone.now)
@admin.register(TrainingWalkStraight)
class TrainingWalkStraightAdmin(admin.ModelAdmin):
    list_display = [field.name for field in TrainingWalkStraight._meta.fields]#python幫填寫欄位"""




class train(models.Model):  #訓練資料
    train_name = models.CharField(max_length=20)  # 訓練的名稱
    train_category = models.CharField(max_length=10)  # 訓練類型的名稱
    train_time = models.DateTimeField(max_length=2, null=True, blank=True)  # 訓練需消耗時間
    run_time = models.DateTimeField(default=timezone.now)
    train_frequency = models.CharField(max_length=2)  # 訓練次數
    def __str__(self):
        return self.train_name
@admin.register(train)
class TrainAdmin(admin.ModelAdmin):
    list_display = [field.name for field in train._meta.fields]#python幫填寫欄位
    list_filter = ('train_category',)#過濾器
    #fields = ['train_time','train_frequency']#顯示可修改欄位 會讓增加的資料變成只能增加可修改的欄位
    search_fields = ['train_name']#顯示可搜尋欄位
    ordering = ('train_time',)#排序從小到大 如果要從大到小要在變數前面單引號裡面加上減號'-....'

class TreatmentContent(models.Model):
    user = models.ForeignKey(CustomUser, on_delete=models.CASCADE, related_name='treatments_received')
    therapist = models.ForeignKey(CustomUser, on_delete=models.CASCADE, related_name='treatments_given')
    training_type = models.TextField()
    difficulty = models.TextField(default='Easy')
    frequency = models.PositiveIntegerField(null=True, blank=True)
    status = models.BooleanField(default=False)
@admin.register(TreatmentContent)
class TreatmentAdmin(admin.ModelAdmin):
    list_display = [field.name for field in TreatmentContent._meta.fields]#python幫填寫欄位
