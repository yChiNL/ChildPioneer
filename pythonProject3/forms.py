from django import forms
from django.contrib.auth.forms import AuthenticationForm, PasswordChangeForm
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from django.db import transaction
#from .models import Student, Therapist, User
from .models import CustomUser


class UserRegistrationForm(UserCreationForm):
    class Meta:
        model = CustomUser
        fields = ('username', 'password1', 'password2','child_name', 'email', 'user_type')

class UserProfileUpdateForm(forms.ModelForm):
    class Meta:
        model = CustomUser
        fields = ['child_name', 'email', 'parent_name', 'parent_phonenumber', 'parent_line']

class ChangePasswordForm(PasswordChangeForm):
    class Meta:
        model = CustomUser
        fields = ('old_password', 'new_password1', 'new_password2')

class TherapistRegistrationForm(UserCreationForm):
    class Meta:
        model = CustomUser
        fields = ('username', 'password1', 'password2', 'therapist_name', 'email', 'user_type')

class TherapistProfileUpdateForm(forms.ModelForm):
    class Meta:
        model = CustomUser
        fields = ['therapist_name', 'email', 'therapist_phonenumber', 'therapist_line']


class RegisterForm(UserCreationForm):
    username = forms.CharField(
        label="帳號",
        widget=forms.TextInput(attrs={'class': 'form-control'})
    )
    email = forms.EmailField(
        label="電子郵件",
        widget=forms.EmailInput(attrs={'class': 'form-control'})
    )
    password1 = forms.CharField(
        label="密碼",
        widget=forms.PasswordInput(attrs={'class': 'form-control'})
    )
    password2 = forms.CharField(
        label="密碼確認",
        widget=forms.PasswordInput(attrs={'class': 'form-control'})
    )
    child_name = forms.CharField(
        label="兒童姓名",
        widget=forms.TextInput(attrs={'class': 'form-control'})
    )
    parent_name = forms.CharField(
        label="家長姓名",
        widget=forms.TextInput(attrs={'class': 'form-control'})
    )
    parent_phone = forms.CharField(
        label="家長電話",
        widget=forms.TextInput(attrs={'class': 'form-control'})
    )

    class Meta:
        model = User
        fields = ('username', 'email', 'password1', 'password2', 'child_name', 'parent_name', 'parent_phone')

class LoginForm(forms.Form):
    username = forms.CharField(
        label="帳號",
        widget=forms.TextInput(attrs={'class': 'form-control'})
    )
    password = forms.CharField(
        label="密碼",
        widget=forms.PasswordInput(attrs={'class': 'form-control'})
    )
