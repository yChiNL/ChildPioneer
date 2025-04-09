import os
import subprocess
from datetime import datetime, timedelta
from django.contrib import auth
from django.contrib.auth.decorators import login_required
from django.contrib.auth.hashers import make_password
from django.contrib.auth.models import User
from django.shortcuts import render, redirect, get_object_or_404
from django.utils.http import urlsafe_base64_encode
from django.contrib import messages
from ChildPioneer.models import CustomUser, TrainingOneFoot, TrainingJump, TrainingWalkStraight, TrainingDance, \
    TreatmentContent,User_BackgroundImage
from ChildPioneer.forms import LoginForm, TherapistProfileUpdateForm
from ChildPioneer import settings
from django.http import JsonResponse, HttpResponse, HttpResponseBadRequest, HttpResponseForbidden, HttpResponseNotFound
from django.views.decorators.csrf import csrf_exempt
import json
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib.auth import login, authenticate, get_user_model, logout
from django.http import HttpResponseRedirect

from .dance_personal import main as dance_Per
from .jump_personal import main as jump_Per
from .walkstraight_personal import main as walkstright_Per
from .onefoot_personal import main as onefoot_Per

from .forms import RegisterForm, ChangePasswordForm
from django.core.mail import send_mail
import base64
import random
from django.contrib.auth.tokens import PasswordResetTokenGenerator
import string
import six
from django.views.generic import CreateView
from .forms import UserRegistrationForm, TherapistRegistrationForm, UserProfileUpdateForm
from .models import CustomUser
from django.contrib.auth import update_session_auth_hash
from .models import BackgroundImage
from django.db.models import Avg, Sum


@login_required
def schedule_treatment(request, user_id):
    user = CustomUser.objects.get(id=user_id)
    therapist = request.user

    if request.method == 'POST':
        training_type = request.POST['training_type']
        difficulty = request.POST['difficulty']
        frequency = int(request.POST['frequency'])
        treatment_content = TreatmentContent.objects.create(user=user, therapist=therapist, difficulty=difficulty, training_type=training_type, frequency=frequency)

        return redirect('therapist_home')  # 重定向到成功页面
    else:
        treatments = TreatmentContent.objects.filter(user=user)
        return render(request, 'schedule_treatment.html', {'user': user,'treatments':treatments})

@login_required()
def dance_view(request, difficulty, frequency, id):
    from ChildPioneer.dance import main as dance_Ther
    if request.method == 'POST':
        id = id
        user = request.user
        difficulty = difficulty
        frequency = frequency
        background = str(User_BackgroundImage.objects.filter(user=user, useornot=True)[0].image_path)
        dance_Ther(user, difficulty, int(frequency), id, background)  # 调用您的Python函数
        return redirect('Therapist_training')
    return render(request, 'dance.html')

@login_required()
def onefoot_therapist_view(request, difficulty, frequency, id):
    from ChildPioneer.onefoot import main as onefoot_Ther
    if request.method == 'POST':
        id = id
        user = request.user
        difficulty = difficulty
        frequency = frequency
        background = str(User_BackgroundImage.objects.filter(user=user, useornot=True)[0].image_path)

        onefoot_Ther(user, difficulty, int(frequency), id, background)  # 调用您的Python函数
        return redirect('Therapist_training')
    return render(request, 'onefoot.html')

@login_required()
def jump_therapist_view(request, difficulty,  frequency, id):
    from ChildPioneer.jump import main as jump_Ther
    if request.method == 'POST':
        id = id
        user = request.user
        difficulty = difficulty
        frequency = frequency
        background = str(User_BackgroundImage.objects.filter(user=user, useornot=True)[0].image_path)

        jump_Ther(user, difficulty, int(frequency), id, background)  # 调用您的Python函数
        return redirect('Therapist_training')
    return render(request, 'jump.html')

@login_required()
def walkstraight_therapist_view(request, difficulty, frequency,id):
    from ChildPioneer.walkstraight import main as walkstraight_Ther
    if request.method == 'POST':
        id = id
        user = request.user
        difficulty = difficulty
        frequency = frequency
        walkstraight_Ther(user, difficulty, id)  # 调用您的Python函数
        return redirect('Therapist_training')
    return render(request, 'walkstraight.html')

def user_login(request):
    if request.method == 'POST':
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')
            user = authenticate(username=username, password=password)
            if user is not None and user.user_type == 'user':
                login(request, user)
                return redirect('user_home')  # 用户首页
            elif user is not None and user.user_type == 'therapist':
                login(request, user)
                return redirect('add_friend')  # 治疗师首页
    else:
        form = AuthenticationForm()
    return render(request, 'login.html', {'form': form})

def register_user(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            user = form.save()

            login(request, user)
            User_BackgroundImage.objects.create(user=user, name='back_winter', image_id=8, useornot=True, image_path='background_images/back_winter.jpg')
            return redirect('login')  # 用户首页
    else:
        initial_data = {'user_type': 'user'}  # 默认用户类型为 "User"
        form = UserRegistrationForm(initial=initial_data)
    return render(request, 'student_register.html', {'form': form})

def register_therapist(request):
    if request.method == 'POST':
        form = TherapistRegistrationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            return redirect('login')  # 治疗师首页
    else:
        initial_data = {'user_type': 'therapist'}  # 默认用户类型为 "User"
        form = TherapistRegistrationForm(initial=initial_data)
    return render(request, 'therapist_register.html', {'form': form})

def media(request):
    return render(request, "resetpasssword.html")

def identity(request):
    return render(request, "identitySelect.html")

def student_reviseprofile(request):
    return render(request, "student_reviseprofile.html")


@login_required
def update_profile(request):
    if request.method == 'POST':
        if request.user.user_type == 'therapist':
            form = TherapistProfileUpdateForm(request.POST, instance=request.user)
        else:
            form = UserProfileUpdateForm(request.POST, instance=request.user)

        if form.is_valid():
            form.save()
            return redirect('myprofile')  # 或者重定向到适当的页面
    else:
        if request.user.user_type == 'therapist':
            form = TherapistProfileUpdateForm(instance=request.user)
        else:
            form = UserProfileUpdateForm(instance=request.user)

    if request.user.user_type == 'therapist':
        return render(request, 'therapist_update_profile.html', {'form': form})
    else:
        return render(request, 'update_profile.html', {'form': form})


@login_required
def change_password(request):
    template_name = 'change_password.html'  # 默认模板
    success_redirect = 'password_change_done'  # 默认密码修改成功页面

    if request.user.user_type == 'therapist':
        template_name = 'change_password_ther.html'  # 如果是治疗师，使用治疗师的模板
        success_redirect = 'password_change_done_ther'  # 设置治疗师密码修改成功页面的重定向

    if request.method == 'POST':
        form = ChangePasswordForm(request.user, request.POST)
        if form.is_valid():
            user = form.save()
            # 更新会话的认证哈希，以确保用户保持登录状态
            update_session_auth_hash(request, user)
            return redirect(success_redirect)  # 根据用户类型重定向到相应的密码修改成功页面
    else:
        form = ChangePasswordForm(request.user)

    return render(request, template_name, {'form': form})

def password_change_done(request):
    return render(request, 'password_change_done.html')

def password_change_done_ther(request):
    return render(request, 'password_change_done_ther.html')

def store(request):
    # 获取用户的 coins 数量
    user = request.user
    user_coins = user.coins

    # 获取可购买的背景图片列表
    background_images = BackgroundImage.objects.all()

    context = {
        'user_coins': user_coins,
        'background_images': background_images,
    }

    return render(request, 'store.html', context)

def purchase_background(request, background_id):
    # 处理购买背景图片的逻辑
    # 获取用户
    user = request.user
    # 获取背景图片
    background_image = BackgroundImage.objects.get(pk=background_id)
    if user.coins >= background_image.price:
        # 如果用户有足够的 coins，扣除 coins 并关联背景图片到用户
        user.coins -= background_image.price
        user.save()
        user.background_images.add(background_image)  # 假设有一个 ManyToManyField 来关联用户和背景图片
        try:
            User_BackgroundImage.objects.get(user=user, image_id=background_id)
        except User_BackgroundImage.DoesNotExist:
            new_background = User_BackgroundImage.objects.create(user=user, image_id=background_id, name=background_image.name, image_path=background_image.image_path)
            new_background.save()
        purchase_success = True  # 设置购买成功标志
    else:
        # 如果用户 coins 不足，重定向回商店页面并显示错误消息
        purchase_success = False

    # 获取可购买的背景图片列表
    background_images = BackgroundImage.objects.all()

    context = {
        'user_coins': user.coins,
        'background_images': background_images,
        'purchase_success': purchase_success,  # 将购买成功标志传递给模板
    }

    return render(request, 'store.html', context)



@login_required
def collections(request):
    user = request.user
    if request.method == 'POST':
        User_BackgroundImage.objects.filter(user=user).update(useornot=False)
        image_id = request.POST.get('image_id')
        User_BackgroundImage.objects.filter(user=user, image_id=image_id).update(useornot=True)

        context = {
            'user': user,  # 将用户对象添加到上下文中
            'background_images': user.background_images.all(),  # 将用户的背景图片添加到上下文中
        }
        return render(request, 'collections.html', context)
    else:
        context = {
            'user': user,  # 将用户对象添加到上下文中
            'background_images': user.background_images.all(),  # 将用户的背景图片添加到上下文中
        }
        return render(request, 'collections.html', context)


def Personal_TrainSelect(request):
    return render(request, 'Personal_TrainSelect.html')

def user_logout(request):
    logout(request)
    return redirect('login')

#walkstraight_personal未做
def dance_SelectMode(request):
    return render(request, 'dance_SelectMode.html')
def dance_start(request, mode):#個人訓練
    selected_mode = request.POST.get('selected_mode', 'VeryEasy')
    mode_descriptions = {
        'VeryEasy': {
            'video': 'https://www.youtube.com/embed/IHyD4RFsMlA?si=WOua-XmMvLAsDyv3',
            'ShowMode': 'Level1',
            'Description': '內容:手部擺動訓練，共包含三個動作，孩童需擺出類似飛機機翼的動作。 ',
            '目標': '目標:需達成動作三次且於六分鐘內完成即算成功。'
        },
        'Easy': {
            'video': 'https://www.youtube.com/embed/mG1eoGdIdQs?si=gmdxHuS3n4aO1Jq2',
            'ShowMode': 'Level2',
            'Description': '內容：共２種動作，主要訓練孩童對於手肘的控制能力。',
            '目標': '目標:需達成動作五次且於八分鐘內完成即算成功。'
        },
        'Normal': {
            'video': 'https://www.youtube.com/embed/HLccULmp_tE?si=ZVZCQ06VaK_p-PIj',
            'ShowMode': 'Level3',
            'Description': '內容：手部不對稱動作訓練，包含2個動作，將初階之對稱動作延伸為不對稱動作。',
            '目標': '目標:需達成動作五次且於十分鐘內完成即算成功。'
        },
        'Hard': {
            'video': 'https://www.youtube.com/embed/QTm9TrVdS0Y?si=1nIQ6bx7HTAhqNig',
            'ShowMode': 'Level4',
            'Description': '內容：肢體協調舞蹈訓練，包含四個動作，此難度整合了初階、中階之訓練動作。',
            '目標': '目標:需達成動作八次且於十二分鐘內完成即算成功。'
        },
        'VeryHard': {
            'video': 'https://www.youtube.com/embed/JlbDNtIDkE0?si=JuYhD53IYbevO09L',
            'ShowMode': 'Level5',
            'Description': '內容:包含四個動作。此難度納入訓練孩童控制手腕的能力。',
            '目標': '目標:需達成動作十次且於十四分鐘內完成即算成功。'
        }
    }
    video = mode_descriptions[mode]['video']
    ShowMode = mode_descriptions[mode]['ShowMode']
    description = mode_descriptions[mode]['Description']
    goal = mode_descriptions[mode]['目標']
    if request.method == 'POST':
        user = request.user
        background = str(User_BackgroundImage.objects.filter(user=user, useornot=True)[0].image_path)
        dance_Per(request.user, mode,background_name=background)  # 呼叫dance_test函數
        # 重新導向到 Personal_TrainSelect 頁面
        return redirect('Personal_TrainSelect')

        # 自訂HTML內容
        custom_html_content = """
        <html>
        <head>
            <title>訓練開始</title>
        </head>
        <body>
            <h1>成功！</h1>
            <p>模式: {mode}</p>
            <p>模式描述: {description}</p>
            <p>目標: {goal}</p>
            <!-- 在這裡添加您的自訂內容 -->
        </body>
        </html>
        """.format(mode=mode, description=mode_descriptions[mode]['Description'], goal=mode_descriptions[mode]['目標'])
        # 返回自訂HTML頁面
        return HttpResponse(custom_html_content)
        # 返回頁面或HttpResponse
        return HttpResponse("開始訓練！")  # 這只是一個示例響應，您可以根據需要返回適當的響應
    return render(request, 'dance_start.html', {'mode': mode,'video':video, 'ShowMode': ShowMode, 'mode_description': description, 'goal': goal })

def walkstraight_view(request):
    user = CustomUser.objects.get(username='stest')  # Replace 'stest' with the actual username
    mode = 'Easy'  # Replace 'Easy' with the desired mode
    main(user, mode)
    return render(request, 'walkstraight.html')
def jump_SelectMode(request):
    return render(request, 'jump_SelectMode.html')
def jump_start(request, mode):#個人訓練
    selected_mode = request.POST.get('selected_mode', 'VeryEasy')
    mode_descriptions = {
        'VeryEasy': {
            'video': 'https://www.youtube.com/embed/gumTMV1wt_k?si=i4qpin9KcCmggb1a',
            'ShowMode': 'Level1',
            'Description': '內容:以小老鼠的圖案作為固定標的物，高度為10公分，孩童需要跳過小老鼠圖案',
            '目標': '目標:需達成動作三次且於六分鐘內完成即算成功。',
        },
        'Easy': {
            'video': 'https://www.youtube.com/embed/qDNUZ8Aik4w?si=NC03aihb0g1BiRvT',
            'ShowMode': 'Level2',
            'Description': '內容:以小貓咪的圖案作為固定標的物，高度為20公分，孩童需要跳過小貓咪圖案',
            '目標': '目標:達成動作五次且於八分鐘內完成即算成功'
        },
        'Normal': {
            'video': 'https://www.youtube.com/embed/AikiwZ9Q20A?si=6fynY-kx6axR3jx8',
            'ShowMode': 'Level3',
            'Description': '內容:以長頸鹿的圖案作為固定標的物，高度為30公分，孩童需要跳過長頸鹿圖案',
            '目標': '目標:達成動作八次且於十分鐘內完成即算成功'
        },
        'Hard': {
            'video': 'https://www.youtube.com/embed/sHYtQ9-XaYA?si=We76QfNPo-iSMK9k',
            'ShowMode': 'Level4',
            'Description': '內容:老鼠將以動態的形式帶領著兒童跳過懸崖，老鼠至對面懸崖後停留在崖上等待兒童，孩童至同一側後再進行下一輪動作',
            '目標': '目標:達成動作八次且於十二分鐘內完成即算成功。'
        },
        'VeryHard': {
            'video': 'https://www.youtube.com/embed/1yec4nTJ6MM?si=Zv1tVSNKVto_6b8l',
            'ShowMode': 'Level5',
            'Description': '內容:在高階的基礎下，增加懸崖之間的寬度，寬度設定為30公分',
            '目標': '目標:達成動作十二次並於十四分鐘內完成即算成功'
        }
    }
    video = mode_descriptions[mode]['video']
    ShowMode = mode_descriptions[mode]['ShowMode']
    description = mode_descriptions[mode]['Description']
    goal = mode_descriptions[mode]['目標']
    if request.method == 'POST':
        user = request.user
        background = str(User_BackgroundImage.objects.filter(user=user, useornot=True)[0].image_path)
        jump_Per(request.user, mode, background_name=background)  # 呼叫jump_per函數
        return redirect('Personal_TrainSelect')
        # 返回頁面或HttpResponse
        return HttpResponse("開始訓練！")  # 這只是一個示例響應，您可以根據需要返回適當的響應
    return render(request, 'jump_start.html', {'mode': mode, 'video':video, 'ShowMode': ShowMode, 'mode_description': description, 'goal': goal })

def onefoot_SelectMode(request):
    return render(request, 'onefoot_SelectMode.html')
def onefoot_start(request, mode):#個人訓練
    selected_mode = request.POST.get('selected_mode', 'VeryEasy')
    mode_descriptions = {
        'VeryEasy_Left': {
            'video': 'https://www.youtube.com/embed/ldOmwZstgYQ?si=Fj0uC0EdeuGBufRd',
            'ShowMode': 'Level1',
            'Description': '內容:維持左單腳站立3秒。',
            '目標': '目標:，達成動作三次並於六分鐘內完成。'
        },
        'VeryEasy_Right': {
            'video': 'https://www.youtube.com/embed/owX4_VZB_f0?si=b4xSUv0fAke0ZWPP',
            'ShowMode': 'Level1',
            'Description': '內容:維持右單腳站立3秒。',
            '目標': '目標:維持右腳單腳站立3秒，達成動作三次並於六分鐘內完成。'
        },
        'Easy_Left': {
            'video': 'https://www.youtube.com/embed/Zi86iMrEwA0?si=TMDg92EoIVlN11VO',
            'ShowMode': 'Level2',
            'Description': '內容:維持左單腳站立3秒。',
            '目標': '目標:維持左腳單腳站立3秒，達成動作五次並於八分鐘內完成。'
        },
        'Easy_Right': {
            'video': 'https://www.youtube.com/embed/IuOnhCUcXPY?si=DhxSVTnU5p1koagC',
            'ShowMode': 'Level2',
            'Description': '內容:維持右單腳站立3秒。',
            '目標': '目標:維持右腳單腳站立3秒，達成動作五次並於八分鐘內完成。'
        },
        'Normal': {
            'video': 'https://www.youtube.com/embed/4_3g48igg_I?si=ZnlONOws1xx7Lwj0',
            'ShowMode': 'Level3',
            'Description': '內容:孩童需維持單腳站立5秒，每成功一次鱷魚會出現於另一隻腳下，孩童需改為將另一隻腳抬起。',
            '目標': '目標:雙腳各達成三次，並於十分鐘內完成。'
        },
        'Hard_Left': {
            'video': 'https://www.youtube.com/embed/Ru3YVpa4De4?si=2YxoHSWfRjUFibJm',
            'ShowMode': 'Level4',
            'Description': '內容:由單腳站立進階為單腳跳躍。左腳需先放置於畫面中的石頭上，在鱷魚出現3秒後，孩童需跳起。',
            '目標': '目標:達成動作三次並於十二分鐘內完成。'
        },
        'Hard_Right': {
            'video': 'https://www.youtube.com/embed/oTX0NmU3D8w?si=vmtDMgqU4VmeyhQj',
            'ShowMode': 'Level4',
            'Description': '內容:由單腳站立進階為單腳跳躍。右腳需先放置於畫面中的石頭上，在鱷魚出現3秒後，孩童需跳起。',
            '目標': '目標:達成動作三次並於十二分鐘內完成。'
        },
    }
    video = mode_descriptions[mode]['video']
    ShowMode = mode_descriptions[mode]['ShowMode']
    description = mode_descriptions[mode]['Description']
    goal = mode_descriptions[mode]['目標']
    if request.method == 'POST':
        user = request.user
        background = str(User_BackgroundImage.objects.filter(user=user, useornot=True)[0].image_path)
        onefoot_Per(request.user, mode, background_name=background)  # 呼叫onefoot_Per函數
        return redirect('Personal_TrainSelect')
        # 返回頁面或HttpResponse
        return HttpResponse("開始訓練！")  # 這只是一個示例響應，您可以根據需要返回適當的響應
    return render(request, 'onefoot_start.html', {'mode': mode, 'video':video, 'ShowMode': ShowMode, 'mode_description': description, 'goal': goal })

def walk_SelectMode(request):
    return render(request, 'walk_SelectMode.html')
def walk_start(request, mode):#個人訓練
    selected_mode = request.POST.get('selected_mode', 'Easy')
    mode_descriptions = {
        'Easy': {
            'video': 'https://www.youtube.com/embed/FKfvLeKQfMA?si=RYF3aGrHTR7TUbsU',
            'ShowMode': 'Level1',
            'Description': '內容:最開始的階段為訓練一般直線行走的能力。',
            '目標': '目標:從起始點行走至終點線即算成功'
        },
        'Normal': {
            'video': 'https://www.youtube.com/embed/4kAA46ivI68?si=7hlyUKeLq3-EPlbh',
            'ShowMode': 'Level2',
            'Description': '內容:場景縮小25 %，並增加少量障礙物。',
            '目標': '目標:從起始點行走至終點線即算成功'
        },
        'Hard': {
            'video': 'https://www.youtube.com/embed/OU3ehdIut-c?si=QN_TzvJlGRb8Fzsb',
            'ShowMode': 'Level3',
            'Description': '內容:場景縮小50 %，並增加較多障礙物。',
            '目標': '目標:從起始點行走至終點線即算成功'
        }
    }
    video = mode_descriptions[mode]['video']
    ShowMode = mode_descriptions[mode]['ShowMode']
    description = mode_descriptions[mode]['Description']
    goal = mode_descriptions[mode]['目標']
    if request.method == 'POST':
        user = request.user

        walkstright_Per(request.user, mode)  # 呼叫jump_per函數
        return redirect('Personal_TrainSelect')
        # 返回頁面或HttpResponse
        return HttpResponse("開始訓練！")  # 這只是一個示例響應，您可以根據需要返回適當的響應
    return render(request, 'walk_start.html', {'mode': mode, 'video':video, 'ShowMode': ShowMode, 'mode_description': description, 'goal': goal })


def starttrain_view(request):
    # 在此處可以添加任何必要的邏輯，例如從數據庫檢索數據等
    # 渲染starttrain.html並將其傳遞給模板
    return render(request, 'starttrain.html')

# def dance_test(request, mode):
#     # Your code to handle the dance_test view
#     return render(request, 'dance_test.html', {'mode': mode})


def add_friend(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        try:
            user_to_add = CustomUser.objects.get(username=username)

            # 只有 User 类型的用户能够添加为好友
            if user_to_add != request.user and user_to_add.user_type == 'user':
                request.user.friends.add(user_to_add)
        except CustomUser.DoesNotExist:
            pass
    return redirect('therapist_home')
def delete_friend(request, friend_id):
    if request.method == 'POST':
        try:
            friend_to_delete = CustomUser.objects.get(id=friend_id)
            if friend_to_delete in request.user.friends.all():
                request.user.friends.remove(friend_to_delete)
        except CustomUser.DoesNotExist:
            pass
    return redirect('therapist_home')

def therapist_home(request):
    if request.user.user_type == 'therapist':
        friends = request.user.friends.all()
        return render(request, 'therapist_home.html', {'friends': friends})
    else:
        # 处理不是 therapist 的用户
        pass

def user_home(request):
        return render(request, 'user_home.html')


class CustomPasswordResetTokenGenerator(PasswordResetTokenGenerator):
    def _make_hash_value(self, user, timestamp):
        return (
            six.text_type(user.pk) + six.text_type(timestamp) +
            six.text_type(user.is_active)
        )

def forgot_password(request):
    if request.method == "GET":
        return render(request, "forgot_password.html")
    else:
        user = request.POST.get("username")
        forgot_user = CustomUser.objects.get(username=user)
        if forgot_user:
            email = forgot_user.email
            email_part = email[3:]

            random_password = ''.join(random.sample(string.ascii_letters + string.digits, 8))
            password = make_password(random_password)
            CustomUser.objects.filter(username=user).update(password=password)

            subject = "Password reset notification"
            message = "Your username " + user + " 's password is changed into " + random_password
            sender = settings.EMAIL_HOST_USER
            recipient = [email]
            send_mail(
                subject,
                message,
                sender,
                recipient
            )

            return render(request, "forgot_password.html", {"forgot_password_tips": "Password is sent to *****" + email_part + " , please check your inbox or junk box."})
        else:
            print(forgot_user)
            return render(request, "forgot_password.html", {"forgot_password_tips": user + " is not exist!"})




def Training_mode(request):
    return render(request, "Training_mode.html", locals())

def therapist_training_view(request):
    if request.user.user_type == 'user':
        user = request.user
        treatments = TreatmentContent.objects.filter(user=user,status=False)  # 获取所有治療內容数据
        return render(request, 'therapist_training.html', {'treatments': treatments})
    else:
        # 处理不是 user 的用户
        pass

def Personal_training(request):
    return render(request, "Personal_training.html")
def Personal_training_jump(request):
    return render(request, "Personal_training_jump.html")
def Personal_training_onefoot(request):
    return render(request, "Personal_training_onefoot.html")
def Personal_training_walk(request):
    return render(request, "Personal_training_walk.html")

def buttontest(request):

    return render(request, 'buttontest.html')

def myprofile(request):
    return render(request, "myprofile.html")

def dataimport(request):
    # 從資料庫中獲取資料，這裡以簡單地選取所有的資料為例
    data = TrainingDance.objects.all()
    return render(request, 'Data.html', {'data': data})

def contact_therapists(request):
    if request.user.is_authenticated and request.user.user_type == 'user':
        user_friends = request.user.friends.all()
        therapists = CustomUser.objects.filter(user_type='therapist', id__in=user_friends)
        return render(request, 'contacttherapist.html', {'therapists': therapists})
    return render(request, 'contacttherapist.html', {'therapists': []})  # 用户未登录或非USER类型的情况下，不显示治疗师

@login_required
def contact_user(request, user_id):
    try:
        user = CustomUser.objects.get(id=user_id)
    except CustomUser.DoesNotExist:
        return redirect('therapist_home')  # 如果用户不存在，重定向到治疗师主页

    return render(request, 'contactusers.html', {'user': user})

@login_required  # 登入保護視圖
def chart_index_ther(request, user_name):
    user = request.user
    if user.user_type == 'therapist':
        print(str(user_name)+" "+str(user))
        if user_name and user_name in [str(friend.child_name) for friend in user.friends.all()]:
            # 在這裡使用 user_id 來獲取指定使用者的訓練紀錄
            user_with_records = CustomUser.objects.get(child_name=user_name)
            user_id = CustomUser.objects.values('id').get(child_name=user_name)['id']
            dance_records = TrainingDance.objects.filter(user=user_with_records)
            onefoot_records = TrainingOneFoot.objects.filter(user=user_with_records)
            jump_records = TrainingJump.objects.filter(user=user_with_records)
            walkstraight_records = TrainingWalkStraight.objects.filter(user=user_with_records)
            return render(request, 'chart_index_ther.html', {
                'user_id': user_id,
                'dance_records': dance_records,
                'onefoot_records': onefoot_records,
                'jump_records': jump_records,
                'walkstraight_records': walkstraight_records,
                'user_name': user_name
            })
        else:
            return HttpResponseForbidden("You are not authorized to access this page.")
    else:
        return render(request, "therapist_home.html")




#下方全為治療紀錄views
@login_required  # 登入保護視圖
def chart_index(request, user_id=None):
    user = request.user
    if user.user_type == 'therapist':
        if user_id and user_id in [str(friend.id) for friend in user.friends.all()]:
            # 在這裡使用 user_id 來獲取指定使用者的訓練紀錄
            user_with_records = CustomUser.objects.get(id=user_id)
            dance_records = TrainingDance.objects.filter(user=user_with_records)
            onefoot_records = TrainingOneFoot.objects.filter(user=user_with_records)
            jump_records = TrainingJump.objects.filter(user=user_with_records)
            walkstraight_records = TrainingWalkStraight.objects.filter(user=user_with_records)
            return render(request, 'chart_index.html', {
                'user_id': user_id,
                'dance_records': dance_records,
                'onefoot_records': onefoot_records,
                'jump_records': jump_records,
                'walkstraight_records': walkstraight_records,
            })
        else:
            return HttpResponseForbidden("You are not authorized to access this page.")
    elif user.user_type == 'user':
        return render(request, 'chart_index.html')
def get_training_data(request):
    user = request.user  # 获取当前登录用户
    very_easy_data = TrainingDance.objects.filter(user=user, difficulty='VeryEasy').count()
    easy_data = TrainingDance.objects.filter(user=user, difficulty='Easy').count()
    normal_data = TrainingDance.objects.filter(user=user, difficulty='Normal').count()
    hard_data = TrainingDance.objects.filter(user=user, difficulty='Hard').count()
    very_hard_data = TrainingDance.objects.filter(user=user, difficulty='VeryHard').count()
    very_easy_data_jump = TrainingJump.objects.filter(user=user, difficulty='VeryEasy').count()
    easy_data_jump = TrainingJump.objects.filter(user=user, difficulty='Easy').count()
    normal_data_jump = TrainingJump.objects.filter(user=user, difficulty='Normal').count()
    hard_data_jump = TrainingJump.objects.filter(user=user, difficulty='Hard').count()
    very_hard_data_jump = TrainingJump.objects.filter(user=user, difficulty='VeryHard').count()
    easy_data_walk = TrainingWalkStraight.objects.filter(user=user, difficulty='Easy').count()
    normal_data_walk = TrainingWalkStraight.objects.filter(user=user, difficulty='Normal').count()
    hard_data_walk = TrainingWalkStraight.objects.filter(user=user, difficulty='Hard').count()
    very_easy_left_data_onefoot = TrainingOneFoot.objects.filter(user=user, difficulty='VeryEasy_Left').count()
    very_easy_right_data_onefoot = TrainingOneFoot.objects.filter(user=user, difficulty='VeryEasy_Right').count()
    easy_left_data_onefoot = TrainingOneFoot.objects.filter(user=user, difficulty='Easy_Left').count()
    easy_right_data_onefoot = TrainingOneFoot.objects.filter(user=user, difficulty='Easy_Right').count()
    normal_data_onefoot = TrainingOneFoot.objects.filter(user=user, difficulty='Normal').count()
    hard_left_data_onefoot = TrainingOneFoot.objects.filter(user=user, difficulty='Hard_Left').count()
    hard_right_data_onefoot = TrainingOneFoot.objects.filter(user=user, difficulty='Hard_Right').count()


    data = {
        'very_easy': very_easy_data,
        'easy': easy_data,
        'normal': normal_data,
        'hard': hard_data,
        'very_hard': very_hard_data,
        'very_easy_jump': very_easy_data_jump,
        'easy_jump': easy_data_jump,
        'normal_jump': normal_data_jump,
        'hard_jump': hard_data_jump,
        'very_hard_jump': very_hard_data_jump,
        'easy_walk' : easy_data_walk,
        'normal_walk': normal_data_walk,
        'hard_walk': hard_data_walk,
        'very_easy_left_onefoot':very_easy_left_data_onefoot,
        'very_easy_right_onefoot':very_easy_right_data_onefoot,
        'easy_left_onefoot':easy_left_data_onefoot,
        'easy_right_onefoot':easy_right_data_onefoot,
        'normal_onefoot':normal_data_onefoot,
        'hard_left_onefoot':hard_left_data_onefoot,
        'hard_right_onefoot':hard_right_data_onefoot,
    }

    return JsonResponse(data)

@login_required  # 登入保護視圖
def chart_index_year(request, user_id=None):
    user = request.user
    if user.user_type == 'therapist':
        if user_id and user_id in [str(friend.id) for friend in user.friends.all()]:
            # 在這裡使用 user_id 來獲取指定使用者的訓練紀錄
            user_with_records = CustomUser.objects.get(id=user_id)

            # 获取当前年份
            current_year = datetime.now().year

            # 使用当前年份进行日期筛选
            dance_records = TrainingDance.objects.filter(user=user_with_records, date__year=current_year)
            onefoot_records = TrainingOneFoot.objects.filter(user=user_with_records, date__year=current_year)
            jump_records = TrainingJump.objects.filter(user=user_with_records, date__year=current_year)
            walkstraight_records = TrainingWalkStraight.objects.filter(user=user_with_records, date__year=current_year)

            return render(request, 'chart_index_year.html', {
                'user_id': user_id,
                'dance_records': dance_records,
                'onefoot_records': onefoot_records,
                'jump_records': jump_records,
                'walkstraight_records': walkstraight_records,
            })
        else:
            return HttpResponseForbidden("You are not authorized to access this page.")
    elif user.user_type == 'user':
        return render(request, 'chart_index_year.html')

@login_required
def get_training_data_dance(request, date):
    user = request.user
    target_date = datetime.strptime(date, "%Y-%m-%d").date()
    # 初始化包含所有月份的字典
    dance = TrainingDance.objects.filter(user=user, run_time__date=target_date).count()
    # 将数据按月份汇总
    data = {
        'dance': dance
    }
    return JsonResponse(data)

@login_required
def get_training_data_jump(request, date):
    user = request.user
    target_date = datetime.strptime(date, "%Y-%m-%d").date()
    # 初始化包含所有月份的字典
    jump = TrainingJump.objects.filter(user=user, run_time__date=target_date).count()
    # 将数据按月份汇总
    data = {
        'jump': jump
    }
    return JsonResponse(data)

@login_required
def get_training_data_walk(request, date):
    user = request.user
    target_date = datetime.strptime(date, "%Y-%m-%d").date()
    # 初始化包含所有月份的字典
    walk = TrainingWalkStraight.objects.filter(user=user, run_time__date=target_date).count()
    # 将数据按月份汇总
    data = {
        'walk': walk
    }
    return JsonResponse(data)


@login_required
def get_training_data_foot(request, date):
    user = request.user
    target_date = datetime.strptime(date, "%Y-%m-%d").date()
    # 初始化包含所有月份的字典
    foot = TrainingOneFoot.objects.filter(user=user, run_time__date=target_date).count()
    # 将数据按月份汇总
    data = {
        'foot': foot
    }
    return JsonResponse(data)


def get_training_data_year(request):
    user = request.user
    current_year = datetime.now().year

    very_easy_data = TrainingDance.objects.filter(user=user, difficulty='VeryEasy', run_time__year=current_year).count()
    easy_data = TrainingDance.objects.filter(user=user, difficulty='Easy', run_time__year=current_year).count()
    normal_data = TrainingDance.objects.filter(user=user, difficulty='Normal', run_time__year=current_year).count()
    hard_data = TrainingDance.objects.filter(user=user, difficulty='Hard', run_time__year=current_year).count()
    very_hard_data = TrainingDance.objects.filter(user=user, difficulty='VeryHard', run_time__year=current_year).count()
    very_easy_data_jump = TrainingJump.objects.filter(user=user, difficulty='VeryEasy', run_time__year=current_year).count()
    easy_data_jump = TrainingJump.objects.filter(user=user, difficulty='Easy', run_time__year=current_year).count()
    normal_data_jump = TrainingJump.objects.filter(user=user, difficulty='Normal', run_time__year=current_year).count()
    hard_data_jump = TrainingJump.objects.filter(user=user, difficulty='Hard', run_time__year=current_year).count()
    very_hard_data_jump = TrainingJump.objects.filter(user=user, difficulty='VeryHard', run_time__year=current_year).count()
    easy_data_walk = TrainingWalkStraight.objects.filter(user=user, difficulty='Easy', run_time__year=current_year).count()
    normal_data_walk = TrainingWalkStraight.objects.filter(user=user, difficulty='Normal', run_time__year=current_year).count()
    hard_data_walk = TrainingWalkStraight.objects.filter(user=user, difficulty='Hard', run_time__year=current_year).count()
    very_easy_left_data_onefoot = TrainingOneFoot.objects.filter(user=user, difficulty='VeryEasy_Left', run_time__year=current_year).count()
    very_easy_right_data_onefoot = TrainingOneFoot.objects.filter(user=user, difficulty='VeryEasy_Right', run_time__year=current_year).count()
    easy_left_data_onefoot = TrainingOneFoot.objects.filter(user=user, difficulty='Easy_Left', run_time__year=current_year).count()
    easy_right_data_onefoot = TrainingOneFoot.objects.filter(user=user, difficulty='Easy_Right', run_time__year=current_year).count()
    normal_data_onefoot = TrainingOneFoot.objects.filter(user=user, difficulty='Normal', run_time__year=current_year).count()
    hard_left_data_onefoot = TrainingOneFoot.objects.filter(user=user, difficulty='Hard_Left', run_time__year=current_year).count()
    hard_right_data_onefoot = TrainingOneFoot.objects.filter(user=user, difficulty='Hard_Right', run_time__year=current_year).count()


    data = {
        'very_easy': very_easy_data,
        'easy': easy_data,
        'normal': normal_data,
        'hard': hard_data,
        'very_hard': very_hard_data,
        'very_easy_jump': very_easy_data_jump,
        'easy_jump': easy_data_jump,
        'normal_jump': normal_data_jump,
        'hard_jump': hard_data_jump,
        'very_hard_jump': very_hard_data_jump,
        'easy_walk': easy_data_walk,
        'normal_walk': normal_data_walk,
        'hard_walk': hard_data_walk,
        'very_easy_left_onefoot': very_easy_left_data_onefoot,
        'very_easy_right_onefoot': very_easy_right_data_onefoot,
        'easy_left_onefoot': easy_left_data_onefoot,
        'easy_right_onefoot': easy_right_data_onefoot,
        'normal_onefoot': normal_data_onefoot,
        'hard_left_onefoot': hard_left_data_onefoot,
        'hard_right_onefoot': hard_right_data_onefoot,
    }

    return JsonResponse(data)



def get_training_data_month(request):
    user = request.user
    current_month = datetime.now().month

    # 使用日期过滤器仅选择当前年份的记录
    very_easy_data = TrainingDance.objects.filter(user=user, difficulty='VeryEasy', run_time__month=current_month).count()
    easy_data = TrainingDance.objects.filter(user=user, difficulty='Easy', run_time__month=current_month).count()
    normal_data = TrainingDance.objects.filter(user=user, difficulty='Normal', run_time__month=current_month).count()
    hard_data = TrainingDance.objects.filter(user=user, difficulty='Hard', run_time__month=current_month).count()
    very_hard_data = TrainingDance.objects.filter(user=user, difficulty='VeryHard', run_time__month=current_month).count()
    very_easy_data_jump = TrainingJump.objects.filter(user=user, difficulty='VeryEasy',run_time__month=current_month).count()
    easy_data_jump = TrainingJump.objects.filter(user=user, difficulty='Easy', run_time__month=current_month).count()
    normal_data_jump = TrainingJump.objects.filter(user=user, difficulty='Normal', run_time__month=current_month).count()
    hard_data_jump = TrainingJump.objects.filter(user=user, difficulty='Hard', run_time__month=current_month).count()
    very_hard_data_jump = TrainingJump.objects.filter(user=user, difficulty='VeryHard', run_time__month=current_month).count()
    easy_data_walk = TrainingWalkStraight.objects.filter(user=user, difficulty='Easy', run_time__month=current_month).count()
    normal_data_walk = TrainingWalkStraight.objects.filter(user=user, difficulty='Normal', run_time__month=current_month).count()
    hard_data_walk = TrainingWalkStraight.objects.filter(user=user, difficulty='Hard', run_time__month=current_month).count()
    very_easy_left_data_onefoot = TrainingOneFoot.objects.filter(user=user, difficulty='VeryEasy_Left', run_time__month=current_month).count()
    very_easy_right_data_onefoot = TrainingOneFoot.objects.filter(user=user, difficulty='VeryEasy_Right', run_time__month=current_month).count()
    easy_left_data_onefoot = TrainingOneFoot.objects.filter(user=user, difficulty='Easy_Left', run_time__month=current_month).count()
    easy_right_data_onefoot = TrainingOneFoot.objects.filter(user=user, difficulty='Easy_Right', run_time__month=current_month).count()
    normal_data_onefoot = TrainingOneFoot.objects.filter(user=user, difficulty='Normal', run_time__month=current_month).count()
    hard_left_data_onefoot = TrainingOneFoot.objects.filter(user=user, difficulty='Hard_Left', run_time__month=current_month).count()
    hard_right_data_onefoot = TrainingOneFoot.objects.filter(user=user, difficulty='Hard_Right', run_time__month=current_month).count()

    data = {
        'very_easy': very_easy_data,
        'easy': easy_data,
        'normal': normal_data,
        'hard': hard_data,
        'very_hard': very_hard_data,
        'very_easy_jump': very_easy_data_jump,
        'easy_jump': easy_data_jump,
        'normal_jump': normal_data_jump,
        'hard_jump': hard_data_jump,
        'very_hard_jump': very_hard_data_jump,
        'easy_walk': easy_data_walk,
        'normal_walk': normal_data_walk,
        'hard_walk': hard_data_walk,
        'very_easy_left_onefoot': very_easy_left_data_onefoot,
        'very_easy_right_onefoot': very_easy_right_data_onefoot,
        'easy_left_onefoot': easy_left_data_onefoot,
        'easy_right_onefoot': easy_right_data_onefoot,
        'normal_onefoot': normal_data_onefoot,
        'hard_left_onefoot': hard_left_data_onefoot,
        'hard_right_onefoot': hard_right_data_onefoot,
    }
    return JsonResponse(data)

def get_training_data_week(request):
    user = request.user
    current_day = datetime.now()
    one_week_ago = current_day - timedelta(days=7)

    # 使用日期过滤器仅选择当前年份的记录
    very_easy_data = TrainingDance.objects.filter(user=user, difficulty='VeryEasy', run_time__range=(one_week_ago, current_day)).count()
    easy_data = TrainingDance.objects.filter(user=user, difficulty='Easy', run_time__range=(one_week_ago, current_day)).count()
    normal_data = TrainingDance.objects.filter(user=user, difficulty='Normal', run_time__range=(one_week_ago, current_day)).count()
    hard_data = TrainingDance.objects.filter(user=user, difficulty='Hard', run_time__range=(one_week_ago, current_day)).count()
    very_hard_data = TrainingDance.objects.filter(user=user, difficulty='VeryHard', run_time__range=(one_week_ago, current_day)).count()
    very_easy_data_jump = TrainingJump.objects.filter(user=user, difficulty='VeryEasy', run_time__range=(one_week_ago, current_day)).count()
    easy_data_jump = TrainingJump.objects.filter(user=user, difficulty='Easy', run_time__range=(one_week_ago, current_day)).count()
    normal_data_jump = TrainingJump.objects.filter(user=user, difficulty='Normal', run_time__range=(one_week_ago, current_day)).count()
    hard_data_jump = TrainingJump.objects.filter(user=user, difficulty='Hard', run_time__range=(one_week_ago, current_day)).count()
    very_hard_data_jump = TrainingJump.objects.filter(user=user, difficulty='VeryHard', run_time__range=(one_week_ago, current_day)).count()
    easy_data_walk = TrainingWalkStraight.objects.filter(user=user, difficulty='Easy', run_time__range=(one_week_ago, current_day)).count()
    normal_data_walk = TrainingWalkStraight.objects.filter(user=user, difficulty='Normal', run_time__range=(one_week_ago, current_day)).count()
    hard_data_walk = TrainingWalkStraight.objects.filter(user=user, difficulty='Hard', run_time__range=(one_week_ago, current_day)).count()
    very_easy_left_data_onefoot = TrainingOneFoot.objects.filter(user=user, difficulty='VeryEasy_Left', run_time__range=(one_week_ago, current_day)).count()
    very_easy_right_data_onefoot = TrainingOneFoot.objects.filter(user=user, difficulty='VeryEasy_Right', run_time__range=(one_week_ago, current_day)).count()
    easy_left_data_onefoot = TrainingOneFoot.objects.filter(user=user, difficulty='Easy_Left', run_time__range=(one_week_ago, current_day)).count()
    easy_right_data_onefoot = TrainingOneFoot.objects.filter(user=user, difficulty='Easy_Right', run_time__range=(one_week_ago, current_day)).count()
    normal_data_onefoot = TrainingOneFoot.objects.filter(user=user, difficulty='Normal', run_time__range=(one_week_ago, current_day)).count()
    hard_left_data_onefoot = TrainingOneFoot.objects.filter(user=user, difficulty='Hard_Left', run_time__range=(one_week_ago, current_day)).count()
    hard_right_data_onefoot = TrainingOneFoot.objects.filter(user=user, difficulty='Hard_Right', run_time__range=(one_week_ago, current_day)).count()

    data = {
        'very_easy': very_easy_data,
        'easy': easy_data,
        'normal': normal_data,
        'hard': hard_data,
        'very_hard': very_hard_data,
        'very_easy_jump': very_easy_data_jump,
        'easy_jump': easy_data_jump,
        'normal_jump': normal_data_jump,
        'hard_jump': hard_data_jump,
        'very_hard_jump': very_hard_data_jump,
        'easy_walk': easy_data_walk,
        'normal_walk': normal_data_walk,
        'hard_walk': hard_data_walk,
        'very_easy_left_onefoot': very_easy_left_data_onefoot,
        'very_easy_right_onefoot': very_easy_right_data_onefoot,
        'easy_left_onefoot': easy_left_data_onefoot,
        'easy_right_onefoot': easy_right_data_onefoot,
        'normal_onefoot': normal_data_onefoot,
        'hard_left_onefoot': hard_left_data_onefoot,
        'hard_right_onefoot': hard_right_data_onefoot,
    }
    return JsonResponse(data)

def get_training_data_day(request):
    user = request.user
    current_day = datetime.now()

    # 使用日期过滤器仅选择当前年份的记录
    very_easy_data = TrainingDance.objects.filter(user=user, difficulty='VeryEasy', run_time__date=current_day).count()
    easy_data = TrainingDance.objects.filter(user=user, difficulty='Easy', run_time__date=current_day).count()
    normal_data = TrainingDance.objects.filter(user=user, difficulty='Normal', run_time__date=current_day).count()
    hard_data = TrainingDance.objects.filter(user=user, difficulty='Hard', run_time__date=current_day).count()
    very_hard_data = TrainingDance.objects.filter(user=user, difficulty='VeryHard', run_time__date=current_day).count()
    very_easy_data_jump = TrainingJump.objects.filter(user=user, difficulty='VeryEasy', run_time__date=current_day).count()
    easy_data_jump = TrainingJump.objects.filter(user=user, difficulty='Easy', run_time__date=current_day).count()
    normal_data_jump = TrainingJump.objects.filter(user=user, difficulty='Normal', run_time__date=current_day).count()
    hard_data_jump = TrainingJump.objects.filter(user=user, difficulty='Hard', run_time__date=current_day).count()
    very_hard_data_jump = TrainingJump.objects.filter(user=user, difficulty='VeryHard', run_time__date=current_day).count()
    easy_data_walk = TrainingWalkStraight.objects.filter(user=user, difficulty='Easy', run_time__date=current_day).count()
    normal_data_walk = TrainingWalkStraight.objects.filter(user=user, difficulty='Normal', run_time__date=current_day).count()
    hard_data_walk = TrainingWalkStraight.objects.filter(user=user, difficulty='Hard', run_time__date=current_day).count()
    very_easy_left_data_onefoot = TrainingOneFoot.objects.filter(user=user, difficulty='VeryEasy_Left', run_time__date=current_day).count()
    very_easy_right_data_onefoot = TrainingOneFoot.objects.filter(user=user, difficulty='VeryEasy_Right', run_time__date=current_day).count()
    easy_left_data_onefoot = TrainingOneFoot.objects.filter(user=user, difficulty='Easy_Left', run_time__date=current_day).count()
    easy_right_data_onefoot = TrainingOneFoot.objects.filter(user=user, difficulty='Easy_Right', run_time__date=current_day).count()
    normal_data_onefoot = TrainingOneFoot.objects.filter(user=user, difficulty='Normal', run_time__date=current_day).count()
    hard_left_data_onefoot = TrainingOneFoot.objects.filter(user=user, difficulty='Hard_Left', run_time__date=current_day).count()
    hard_right_data_onefoot = TrainingOneFoot.objects.filter(user=user, difficulty='Hard_Right', run_time__date=current_day).count()

    data = {
        'very_easy': very_easy_data,
        'easy': easy_data,
        'normal': normal_data,
        'hard': hard_data,
        'very_hard': very_hard_data,
        'very_easy_jump': very_easy_data_jump,
        'easy_jump': easy_data_jump,
        'normal_jump': normal_data_jump,
        'hard_jump': hard_data_jump,
        'very_hard_jump': very_hard_data_jump,
        'easy_walk': easy_data_walk,
        'normal_walk': normal_data_walk,
        'hard_walk': hard_data_walk,
        'very_easy_left_onefoot': very_easy_left_data_onefoot,
        'very_easy_right_onefoot': very_easy_right_data_onefoot,
        'easy_left_onefoot': easy_left_data_onefoot,
        'easy_right_onefoot': easy_right_data_onefoot,
        'normal_onefoot': normal_data_onefoot,
        'hard_left_onefoot': hard_left_data_onefoot,
        'hard_right_onefoot': hard_right_data_onefoot,
    }
    return JsonResponse(data)


#治療師端總表-年
def get_training_data_year_ther(request, user_name):
    print("User name received:", user_name)
    current_year = datetime.now().year
    user = get_object_or_404(CustomUser, child_name=user_name)
    very_easy_data = TrainingDance.objects.filter(user=user, difficulty='VeryEasy', run_time__year=current_year).count()
    easy_data = TrainingDance.objects.filter(user=user, difficulty='Easy', run_time__year=current_year).count()
    normal_data = TrainingDance.objects.filter(user=user, difficulty='Normal', run_time__year=current_year).count()
    hard_data = TrainingDance.objects.filter(user=user, difficulty='Hard', run_time__year=current_year).count()
    very_hard_data = TrainingDance.objects.filter(user=user, difficulty='VeryHard', run_time__year=current_year).count()
    very_easy_data_jump = TrainingJump.objects.filter(user=user, difficulty='VeryEasy',run_time__year=current_year).count()
    easy_data_jump = TrainingJump.objects.filter(user=user, difficulty='Easy', run_time__year=current_year).count()
    normal_data_jump = TrainingJump.objects.filter(user=user, difficulty='Normal', run_time__year=current_year).count()
    hard_data_jump = TrainingJump.objects.filter(user=user, difficulty='Hard', run_time__year=current_year).count()
    very_hard_data_jump = TrainingJump.objects.filter(user=user, difficulty='VeryHard',run_time__year=current_year).count()
    easy_data_walk = TrainingWalkStraight.objects.filter(user=user, difficulty='Easy',run_time__year=current_year).count()
    normal_data_walk = TrainingWalkStraight.objects.filter(user=user, difficulty='Normal',run_time__year=current_year).count()
    hard_data_walk = TrainingWalkStraight.objects.filter(user=user, difficulty='Hard',run_time__year=current_year).count()
    very_easy_left_data_onefoot = TrainingOneFoot.objects.filter(user=user, difficulty='VeryEasy_Left',run_time__year=current_year).count()
    very_easy_right_data_onefoot = TrainingOneFoot.objects.filter(user=user, difficulty='VeryEasy_Right',run_time__year=current_year).count()
    easy_left_data_onefoot = TrainingOneFoot.objects.filter(user=user, difficulty='Easy_Left',run_time__year=current_year).count()
    easy_right_data_onefoot = TrainingOneFoot.objects.filter(user=user, difficulty='Easy_Right',run_time__year=current_year).count()
    normal_data_onefoot = TrainingOneFoot.objects.filter(user=user, difficulty='Normal',run_time__year=current_year).count()
    hard_left_data_onefoot = TrainingOneFoot.objects.filter(user=user, difficulty='Hard_Left',run_time__year=current_year).count()
    hard_right_data_onefoot = TrainingOneFoot.objects.filter(user=user, difficulty='Hard_Right',run_time__year=current_year).count()

    data = {
        'very_easy': very_easy_data,
        'easy': easy_data,
        'normal': normal_data,
        'hard': hard_data,
        'very_hard': very_hard_data,
        'very_easy_jump': very_easy_data_jump,
        'easy_jump': easy_data_jump,
        'normal_jump': normal_data_jump,
        'hard_jump': hard_data_jump,
        'very_hard_jump': very_hard_data_jump,
        'easy_walk': easy_data_walk,
        'normal_walk': normal_data_walk,
        'hard_walk': hard_data_walk,
        'very_easy_left_onefoot': very_easy_left_data_onefoot,
        'very_easy_right_onefoot': very_easy_right_data_onefoot,
        'easy_left_onefoot': easy_left_data_onefoot,
        'easy_right_onefoot': easy_right_data_onefoot,
        'normal_onefoot': normal_data_onefoot,
        'hard_left_onefoot': hard_left_data_onefoot,
        'hard_right_onefoot': hard_right_data_onefoot,
    }

    return JsonResponse(data)


#治療師端總表-月
def get_training_data_month_ther(request, user_name):
    print("User name received:", user_name)
    current_month = datetime.now().month
    user = get_object_or_404(CustomUser, child_name=user_name)
    very_easy_data = TrainingDance.objects.filter(user=user, difficulty='VeryEasy',run_time__month=current_month).count()
    easy_data = TrainingDance.objects.filter(user=user, difficulty='Easy', run_time__month=current_month).count()
    normal_data = TrainingDance.objects.filter(user=user, difficulty='Normal', run_time__month=current_month).count()
    hard_data = TrainingDance.objects.filter(user=user, difficulty='Hard', run_time__month=current_month).count()
    very_hard_data = TrainingDance.objects.filter(user=user, difficulty='VeryHard',run_time__month=current_month).count()
    very_easy_data_jump = TrainingJump.objects.filter(user=user, difficulty='VeryEasy',run_time__month=current_month).count()
    easy_data_jump = TrainingJump.objects.filter(user=user, difficulty='Easy', run_time__month=current_month).count()
    normal_data_jump = TrainingJump.objects.filter(user=user, difficulty='Normal',run_time__month=current_month).count()
    hard_data_jump = TrainingJump.objects.filter(user=user, difficulty='Hard', run_time__month=current_month).count()
    very_hard_data_jump = TrainingJump.objects.filter(user=user, difficulty='VeryHard',run_time__month=current_month).count()
    easy_data_walk = TrainingWalkStraight.objects.filter(user=user, difficulty='Easy',run_time__month=current_month).count()
    normal_data_walk = TrainingWalkStraight.objects.filter(user=user, difficulty='Normal',run_time__month=current_month).count()
    hard_data_walk = TrainingWalkStraight.objects.filter(user=user, difficulty='Hard',run_time__month=current_month).count()
    very_easy_left_data_onefoot = TrainingOneFoot.objects.filter(user=user, difficulty='VeryEasy_Left',run_time__month=current_month).count()
    very_easy_right_data_onefoot = TrainingOneFoot.objects.filter(user=user, difficulty='VeryEasy_Right',run_time__month=current_month).count()
    easy_left_data_onefoot = TrainingOneFoot.objects.filter(user=user, difficulty='Easy_Left',run_time__month=current_month).count()
    easy_right_data_onefoot = TrainingOneFoot.objects.filter(user=user, difficulty='Easy_Right',run_time__month=current_month).count()
    normal_data_onefoot = TrainingOneFoot.objects.filter(user=user, difficulty='Normal',run_time__month=current_month).count()
    hard_left_data_onefoot = TrainingOneFoot.objects.filter(user=user, difficulty='Hard_Left',run_time__month=current_month).count()
    hard_right_data_onefoot = TrainingOneFoot.objects.filter(user=user, difficulty='Hard_Right',run_time__month=current_month).count()

    data = {
        'very_easy': very_easy_data,
        'easy': easy_data,
        'normal': normal_data,
        'hard': hard_data,
        'very_hard': very_hard_data,
        'very_easy_jump': very_easy_data_jump,
        'easy_jump': easy_data_jump,
        'normal_jump': normal_data_jump,
        'hard_jump': hard_data_jump,
        'very_hard_jump': very_hard_data_jump,
        'easy_walk': easy_data_walk,
        'normal_walk': normal_data_walk,
        'hard_walk': hard_data_walk,
        'very_easy_left_onefoot': very_easy_left_data_onefoot,
        'very_easy_right_onefoot': very_easy_right_data_onefoot,
        'easy_left_onefoot': easy_left_data_onefoot,
        'easy_right_onefoot': easy_right_data_onefoot,
        'normal_onefoot': normal_data_onefoot,
        'hard_left_onefoot': hard_left_data_onefoot,
        'hard_right_onefoot': hard_right_data_onefoot,
    }

    return JsonResponse(data)


#治療師端總表-近七天
def get_training_data_week_ther(request, user_name):
    print("User name received:", user_name)
    user = get_object_or_404(CustomUser, child_name=user_name)
    current_month = datetime.now().month
    current_day = datetime.now()
    one_week_ago = current_day - timedelta(days=7)

    very_easy_data = TrainingDance.objects.filter(user=user, difficulty='VeryEasy',run_time__range=(one_week_ago, current_day)).count()
    easy_data = TrainingDance.objects.filter(user=user, difficulty='Easy',run_time__range=(one_week_ago, current_day)).count()
    normal_data = TrainingDance.objects.filter(user=user, difficulty='Normal',run_time__range=(one_week_ago, current_day)).count()
    hard_data = TrainingDance.objects.filter(user=user, difficulty='Hard',run_time__range=(one_week_ago, current_day)).count()
    very_hard_data = TrainingDance.objects.filter(user=user, difficulty='VeryHard',run_time__range=(one_week_ago, current_day)).count()
    very_easy_data_jump = TrainingJump.objects.filter(user=user, difficulty='VeryEasy',run_time__range=(one_week_ago, current_day)).count()
    easy_data_jump = TrainingJump.objects.filter(user=user, difficulty='Easy',run_time__range=(one_week_ago, current_day)).count()
    normal_data_jump = TrainingJump.objects.filter(user=user, difficulty='Normal',run_time__range=(one_week_ago, current_day)).count()
    hard_data_jump = TrainingJump.objects.filter(user=user, difficulty='Hard',run_time__range=(one_week_ago, current_day)).count()
    very_hard_data_jump = TrainingJump.objects.filter(user=user, difficulty='VeryHard',run_time__range=(one_week_ago, current_day)).count()
    easy_data_walk = TrainingWalkStraight.objects.filter(user=user, difficulty='Easy',run_time__range=(one_week_ago, current_day)).count()
    normal_data_walk = TrainingWalkStraight.objects.filter(user=user, difficulty='Normal',run_time__range=(one_week_ago, current_day)).count()
    hard_data_walk = TrainingWalkStraight.objects.filter(user=user, difficulty='Hard',run_time__range=(one_week_ago, current_day)).count()
    very_easy_left_data_onefoot = TrainingOneFoot.objects.filter(user=user, difficulty='VeryEasy_Left',run_time__range=(one_week_ago, current_day)).count()
    very_easy_right_data_onefoot = TrainingOneFoot.objects.filter(user=user, difficulty='VeryEasy_Right',run_time__range=(one_week_ago, current_day)).count()
    easy_left_data_onefoot = TrainingOneFoot.objects.filter(user=user, difficulty='Easy_Left',run_time__range=(one_week_ago, current_day)).count()
    easy_right_data_onefoot = TrainingOneFoot.objects.filter(user=user, difficulty='Easy_Right',run_time__range=(one_week_ago, current_day)).count()
    normal_data_onefoot = TrainingOneFoot.objects.filter(user=user, difficulty='Normal',run_time__range=(one_week_ago, current_day)).count()
    hard_left_data_onefoot = TrainingOneFoot.objects.filter(user=user, difficulty='Hard_Left',run_time__range=(one_week_ago, current_day)).count()
    hard_right_data_onefoot = TrainingOneFoot.objects.filter(user=user, difficulty='Hard_Right',run_time__range=(one_week_ago, current_day)).count()

    data = {
        'very_easy': very_easy_data,
        'easy': easy_data,
        'normal': normal_data,
        'hard': hard_data,
        'very_hard': very_hard_data,
        'very_easy_jump': very_easy_data_jump,
        'easy_jump': easy_data_jump,
        'normal_jump': normal_data_jump,
        'hard_jump': hard_data_jump,
        'very_hard_jump': very_hard_data_jump,
        'easy_walk': easy_data_walk,
        'normal_walk': normal_data_walk,
        'hard_walk': hard_data_walk,
        'very_easy_left_onefoot': very_easy_left_data_onefoot,
        'very_easy_right_onefoot': very_easy_right_data_onefoot,
        'easy_left_onefoot': easy_left_data_onefoot,
        'easy_right_onefoot': easy_right_data_onefoot,
        'normal_onefoot': normal_data_onefoot,
        'hard_left_onefoot': hard_left_data_onefoot,
        'hard_right_onefoot': hard_right_data_onefoot,
    }

    return JsonResponse(data)


#治療師端總表-日
def get_training_data_day_ther(request, user_name):
    print("User name received:", user_name)
    user = get_object_or_404(CustomUser, child_name=user_name)
    current_day = datetime.now()

    very_easy_data = TrainingDance.objects.filter(user=user, difficulty='VeryEasy', run_time__date=current_day).count()
    easy_data = TrainingDance.objects.filter(user=user, difficulty='Easy', run_time__date=current_day).count()
    normal_data = TrainingDance.objects.filter(user=user, difficulty='Normal', run_time__date=current_day).count()
    hard_data = TrainingDance.objects.filter(user=user, difficulty='Hard', run_time__date=current_day).count()
    very_hard_data = TrainingDance.objects.filter(user=user, difficulty='VeryHard', run_time__date=current_day).count()
    very_easy_data_jump = TrainingJump.objects.filter(user=user, difficulty='VeryEasy',run_time__date=current_day).count()
    easy_data_jump = TrainingJump.objects.filter(user=user, difficulty='Easy', run_time__date=current_day).count()
    normal_data_jump = TrainingJump.objects.filter(user=user, difficulty='Normal', run_time__date=current_day).count()
    hard_data_jump = TrainingJump.objects.filter(user=user, difficulty='Hard', run_time__date=current_day).count()
    very_hard_data_jump = TrainingJump.objects.filter(user=user, difficulty='VeryHard',run_time__date=current_day).count()
    easy_data_walk = TrainingWalkStraight.objects.filter(user=user, difficulty='Easy',run_time__date=current_day).count()
    normal_data_walk = TrainingWalkStraight.objects.filter(user=user, difficulty='Normal',run_time__date=current_day).count()
    hard_data_walk = TrainingWalkStraight.objects.filter(user=user, difficulty='Hard',run_time__date=current_day).count()
    very_easy_left_data_onefoot = TrainingOneFoot.objects.filter(user=user, difficulty='VeryEasy_Left',run_time__date=current_day).count()
    very_easy_right_data_onefoot = TrainingOneFoot.objects.filter(user=user, difficulty='VeryEasy_Right',run_time__date=current_day).count()
    easy_left_data_onefoot = TrainingOneFoot.objects.filter(user=user, difficulty='Easy_Left',run_time__date=current_day).count()
    easy_right_data_onefoot = TrainingOneFoot.objects.filter(user=user, difficulty='Easy_Right',run_time__date=current_day).count()
    normal_data_onefoot = TrainingOneFoot.objects.filter(user=user, difficulty='Normal',run_time__date=current_day).count()
    hard_left_data_onefoot = TrainingOneFoot.objects.filter(user=user, difficulty='Hard_Left',run_time__date=current_day).count()
    hard_right_data_onefoot = TrainingOneFoot.objects.filter(user=user, difficulty='Hard_Right',run_time__date=current_day).count()

    data = {
        'very_easy': very_easy_data,
        'easy': easy_data,
        'normal': normal_data,
        'hard': hard_data,
        'very_hard': very_hard_data,
        'very_easy_jump': very_easy_data_jump,
        'easy_jump': easy_data_jump,
        'normal_jump': normal_data_jump,
        'hard_jump': hard_data_jump,
        'very_hard_jump': very_hard_data_jump,
        'easy_walk': easy_data_walk,
        'normal_walk': normal_data_walk,
        'hard_walk': hard_data_walk,
        'very_easy_left_onefoot': very_easy_left_data_onefoot,
        'very_easy_right_onefoot': very_easy_right_data_onefoot,
        'easy_left_onefoot': easy_left_data_onefoot,
        'easy_right_onefoot': easy_right_data_onefoot,
        'normal_onefoot': normal_data_onefoot,
        'hard_left_onefoot': hard_left_data_onefoot,
        'hard_right_onefoot': hard_right_data_onefoot,
    }

    return JsonResponse(data)


@login_required
def get_dance_training_data(request):
    # 获取当前登录用户
    current_user = request.user
    # 获取当前用户的跳舞训练数据，这里使用适当的过滤条件来获取用户自己的数据
    dance_data = TrainingDance.objects.filter(user=current_user)  # 假设有一个与用户关联的字段user用来过滤数据
    # 初始化包含所有月份的字典
    monthly_data = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0}
    # 将数据按月份汇总
    for record in dance_data:
        month = record.run_time.month  # 获取记录的月份
        monthly_data[month] += record.move_count  # 累加每个月的数据

    return JsonResponse(monthly_data)


@login_required
def get_dance_training_data_ther(request, user_name, date):  # 添加 user_name 参数
    print("User name received:", user_name)  # 添加这行代码
    # 获取当前登录的治疗师用户
    user = request.user
    if user.user_type == 'therapist':
        # 使用 get_object_or_404 获取与用户名匹配的用户对象
        user_with_records = get_object_or_404(CustomUser, child_name=user_name)
        target_date = datetime.strptime(date, "%Y-%m-%d").date()
        # 获取指定用户的跳舞训练数据，这里使用适当的过滤条件来获取用户的数据
        dance = TrainingDance.objects.filter(user=user_with_records, run_time__date=target_date).count()
        # 将数据按月份汇总
        data = {
            'dance': dance
        }
        return JsonResponse(data)
    else:
        return HttpResponseForbidden("You are not authorized to access this page.")

@login_required
def get_jump_training_data(request):
    # 获取当前登录用户
    current_user = request.user

    # 获取当前用户的跳跃训练数据，这里使用适当的过滤条件来获取用户自己的数据
    jump_data = TrainingJump.objects.filter(user=current_user)  # 假设有一个与用户关联的字段user用来过滤数据

    # 初始化包含所有月份的字典
    monthly_data = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0}

    # 将数据按月份汇总
    for record in jump_data:
        month = record.run_time.month  # 获取记录的月份
        monthly_data[month] += record.move_count  # 累加每个月的数据

    return JsonResponse(monthly_data)

@login_required
def get_jump_training_data_ther(request, user_name, date):
    print("User name received:", user_name)  # 添加这行代码
    # 获取当前登录的治疗师用户
    user = request.user
    if user.user_type == 'therapist':
        # 使用 get_object_or_404 获取与用户名匹配的用户对象
        user_with_records = get_object_or_404(CustomUser, child_name=user_name)
        target_date = datetime.strptime(date, "%Y-%m-%d").date()
        # 获取指定用户的跳舞训练数据，这里使用适当的过滤条件来获取用户的数据
        jump = TrainingJump.objects.filter(user=user_with_records, run_time__date=target_date).count()
        # 将数据按月份汇总
        data = {
            'jump': jump
        }
        return JsonResponse(data)
    else:
        return HttpResponseForbidden("You are not authorized to access this page.")

@login_required
def get_onefoot_training_data(request):
    # 获取当前登录用户
    current_user = request.user

    # 获取当前用户的单脚训练数据，这里使用适当的过滤条件来获取用户自己的数据
    onefoot_data = TrainingOneFoot.objects.filter(user=current_user)  # 假设有一个与用户关联的字段user用来过滤数据

    # 初始化包含所有月份的字典
    monthly_data = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0}

    # 将数据按月份汇总
    for record in onefoot_data:
        month = record.run_time.month  # 获取记录的月份
        monthly_data[month] += record.move_count  # 累加每个月的数据

    return JsonResponse(monthly_data)

@login_required
def get_onefoot_training_data_ther(request, user_name, date):
    print("User name received:", user_name)  # 添加这行代码
    # 获取当前登录的治疗师用户
    user = request.user
    if user.user_type == 'therapist':
        # 使用 get_object_or_404 获取与用户名匹配的用户对象
        user_with_records = get_object_or_404(CustomUser, child_name=user_name)
        target_date = datetime.strptime(date, "%Y-%m-%d").date()
        # 获取指定用户的跳舞训练数据，这里使用适当的过滤条件来获取用户的数据
        foot = TrainingOneFoot.objects.filter(user=user_with_records, run_time__date=target_date).count()
        # 将数据按月份汇总
        data = {
            'foot': foot
        }
        return JsonResponse(data)
    else:
        return HttpResponseForbidden("You are not authorized to access this page.")

@login_required
def get_walk_training_data(request):
    # 获取当前登录用户
    current_user = request.user

    # 获取当前用户的行走训练数据，这里使用适当的过滤条件来获取用户自己的数据
    walk_data = TrainingWalkStraight.objects.filter(user=current_user)  # 假设有一个与用户关联的字段user用来过滤数据

    # 初始化包含所有月份的字典
    monthly_data = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0}

    # 将数据按月份汇总
    for record in walk_data:
        month = record.run_time.month  # 获取记录的月份
        monthly_data[month] += record.move_count  # 累加每个月的数据

    return JsonResponse(monthly_data)

@login_required
def get_walk_training_data_ther(request, user_name, date):
    print("User name received:", user_name)  # 添加这行代码
    # 获取当前登录的治疗师用户
    user = request.user
    if user.user_type == 'therapist':
        # 使用 get_object_or_404 获取与用户名匹配的用户对象
        user_with_records = get_object_or_404(CustomUser, child_name=user_name)
        target_date = datetime.strptime(date, "%Y-%m-%d").date()
        # 获取指定用户的跳舞训练数据，这里使用适当的过滤条件来获取用户的数据
        walk = TrainingWalkStraight.objects.filter(user=user_with_records, run_time__date=target_date).count()
        # 将数据按月份汇总
        data = {
            'walk': walk
        }
        return JsonResponse(data)
    else:
        return HttpResponseForbidden("You are not authorized to access this page.")

@login_required
def chart_dance(request, user_id=None):
    # 获取当前登录用户
    current_user = request.user

    # 获取当前用户的跳舞训练数据，这里使用适当的过滤条件来获取用户自己的数据
    treatments = TrainingDance.objects.filter(user=current_user).order_by('-run_time')

    very_easy_count = TrainingDance.objects.filter(user=current_user, difficulty='VeryEasy').count()
    easy_count = TrainingDance.objects.filter(user=current_user, difficulty='Easy').count()
    normal_count = TrainingDance.objects.filter(user=current_user, difficulty='Normal').count()
    hard_count = TrainingDance.objects.filter(user=current_user, difficulty='Hard').count()
    very_hard_count = TrainingDance.objects.filter(user=current_user, difficulty='VeryHard').count()

    # 计算各项难度级别的平均训练时长
    very_easy_avg_time = TrainingDance.objects.filter(user=current_user, difficulty='VeryEasy').aggregate(avg_time=Avg('t_time'))['avg_time']
    very_easy_avg_time = round(very_easy_avg_time, 1) if very_easy_avg_time is not None else 0.0

    easy_avg_time = TrainingDance.objects.filter(user=current_user, difficulty='Easy').aggregate(avg_time=Avg('t_time'))['avg_time']
    easy_avg_time = round(easy_avg_time, 1) if easy_avg_time is not None else 0.0

    normal_avg_time = TrainingDance.objects.filter(user=current_user, difficulty='Normal').aggregate(avg_time=Avg('t_time'))['avg_time']
    normal_avg_time = round(normal_avg_time, 1) if normal_avg_time is not None else 0.0

    hard_avg_time = TrainingDance.objects.filter(user=current_user, difficulty='Hard').aggregate(avg_time=Avg('t_time'))['avg_time']
    hard_avg_time = round(hard_avg_time, 1) if hard_avg_time is not None else 0.0

    very_hard_avg_time = TrainingDance.objects.filter(user=current_user, difficulty='VeryHard').aggregate(avg_time=Avg('t_time'))['avg_time']
    very_hard_avg_time = round(very_hard_avg_time, 1) if very_hard_avg_time is not None else 0.0

    # 将记录数量数据放入上下文中，以便在HTML模板中使用
    context = {
        'treatments': treatments,
        'VeryEasy': very_easy_count,
        'Easy': easy_count,
        'Normal': normal_count,
        'Hard': hard_count,
        'VeryHard': very_hard_count,
        'VeryEasy_time': very_easy_avg_time,
        'Easy_time': easy_avg_time,
        'Normal_time': normal_avg_time,
        'Hard_time': hard_avg_time,
        'VeryHard_time': very_hard_avg_time,
    }

    # 返回HTML页面渲染结果
    return render(request, 'chart_dance.html', context)

@login_required
def chart_dance_ther(request, user_name):
    print("User name received:", user_name)  # 添加这行代码
    # 获取当前登录的治疗师用户
    user = request.user
    if user.user_type == 'therapist':
        # 使用 get_object_or_404 获取与用户名匹配的用户对象
        user_with_records = get_object_or_404(CustomUser, child_name=user_name)

        # 获取当前用户的跳舞训练数据，这里使用适当的过滤条件来获取用户自己的数据
        treatments = TrainingDance.objects.filter(user=user_with_records).order_by('-run_time')

        very_easy_count = TrainingDance.objects.filter(user=user_with_records, difficulty='VeryEasy').count()
        easy_count = TrainingDance.objects.filter(user=user_with_records, difficulty='Easy').count()
        normal_count = TrainingDance.objects.filter(user=user_with_records, difficulty='Normal').count()
        hard_count = TrainingDance.objects.filter(user=user_with_records, difficulty='Hard').count()
        very_hard_count = TrainingDance.objects.filter(user=user_with_records, difficulty='VeryHard').count()

        # 计算各项难度级别的平均训练时长
        very_easy_avg_time = TrainingDance.objects.filter(user=user_with_records, difficulty='VeryEasy').aggregate(avg_time=Avg('t_time'))['avg_time']
        very_easy_avg_time = round(very_easy_avg_time, 1) if very_easy_avg_time is not None else 0.0

        easy_avg_time = TrainingDance.objects.filter(user=user_with_records, difficulty='Easy').aggregate(avg_time=Avg('t_time'))['avg_time']
        easy_avg_time = round(easy_avg_time, 1) if easy_avg_time is not None else 0.0

        normal_avg_time = TrainingDance.objects.filter(user=user_with_records, difficulty='Normal').aggregate(avg_time=Avg('t_time'))['avg_time']
        normal_avg_time = round(normal_avg_time, 1) if normal_avg_time is not None else 0.0

        hard_avg_time = TrainingDance.objects.filter(user=user_with_records, difficulty='Hard').aggregate(avg_time=Avg('t_time'))['avg_time']
        hard_avg_time = round(hard_avg_time, 1) if hard_avg_time is not None else 0.0

        very_hard_avg_time = TrainingDance.objects.filter(user=user_with_records, difficulty='VeryHard').aggregate(avg_time=Avg('t_time'))['avg_time']
        very_hard_avg_time = round(very_hard_avg_time, 1) if very_hard_avg_time is not None else 0.0

        # 将记录数量数据放入上下文中，以便在HTML模板中使用
        context = {
            'user_name': user_name,
            'treatments': treatments,
            'VeryEasy': very_easy_count,
            'Easy': easy_count,
            'Normal': normal_count,
            'Hard': hard_count,
            'VeryHard': very_hard_count,
            'VeryEasy_time': very_easy_avg_time,
            'Easy_time': easy_avg_time,
            'Normal_time': normal_avg_time,
            'Hard_time': hard_avg_time,
            'VeryHard_time': very_hard_avg_time,
        }

        # 返回HTML页面渲染结果
        return render(request, 'chart_dance_ther.html', context)
    else:
        return HttpResponseForbidden("You are not authorized to access this page.")

@login_required
def chart_jump(request, user_id=None):
    # 获取当前登录用户
    current_user = request.user

    # 获取当前用户的跳跃训练数据，这里使用适当的过滤条件来获取用户自己的数据
    treatments = TrainingJump.objects.filter(user=current_user).order_by('-run_time')

    very_easy_count = TrainingJump.objects.filter(user=current_user, difficulty='VeryEasy').count()
    easy_count = TrainingJump.objects.filter(user=current_user, difficulty='Easy').count()
    normal_count = TrainingJump.objects.filter(user=current_user, difficulty='Normal').count()
    hard_count = TrainingJump.objects.filter(user=current_user, difficulty='Hard').count()
    very_hard_count = TrainingJump.objects.filter(user=current_user, difficulty='VeryHard').count()

    # 计算各项难度级别的平均训练时长
    very_easy_avg_time = TrainingJump.objects.filter(user=current_user, difficulty='VeryEasy').aggregate(avg_time=Avg('t_time'))['avg_time']
    very_easy_avg_time = round(very_easy_avg_time, 1) if very_easy_avg_time is not None else 0.0

    easy_avg_time = TrainingJump.objects.filter(user=current_user, difficulty='Easy').aggregate(avg_time=Avg('t_time'))['avg_time']
    easy_avg_time = round(easy_avg_time, 1) if easy_avg_time is not None else 0.0

    normal_avg_time = TrainingJump.objects.filter(user=current_user, difficulty='Normal').aggregate(avg_time=Avg('t_time'))['avg_time']
    normal_avg_time = round(normal_avg_time, 1) if normal_avg_time is not None else 0.0

    hard_avg_time = TrainingJump.objects.filter(user=current_user, difficulty='Hard').aggregate(avg_time=Avg('t_time'))['avg_time']
    hard_avg_time = round(hard_avg_time, 1) if hard_avg_time is not None else 0.0

    very_hard_avg_time = TrainingJump.objects.filter(user=current_user, difficulty='VeryHard').aggregate(avg_time=Avg('t_time'))['avg_time']
    very_hard_avg_time = round(very_hard_avg_time, 1) if very_hard_avg_time is not None else 0.0

    # 将记录数量数据放入上下文中，以便在HTML模板中使用
    context = {
        'treatments': treatments,
        'VeryEasy': very_easy_count,
        'Easy': easy_count,
        'Normal': normal_count,
        'Hard': hard_count,
        'VeryHard': very_hard_count,
        'VeryEasy_time': very_easy_avg_time,
        'Easy_time': easy_avg_time,
        'Normal_time': normal_avg_time,
        'Hard_time': hard_avg_time,
        'VeryHard_time': very_hard_avg_time,
    }

    # 返回HTML页面渲染结果
    return render(request, 'chart_jump.html', context)

@login_required
def chart_jump_ther(request, user_name):
    print("User name received:", user_name)  # 添加这行代码
    # 获取当前登录的治疗师用户
    user = request.user
    if user.user_type == 'therapist':
        # 使用 get_object_or_404 获取与用户名匹配的用户对象
        user_with_records = get_object_or_404(CustomUser, child_name=user_name)

        # 获取当前用户的跳舞训练数据，这里使用适当的过滤条件来获取用户自己的数据
        treatments = TrainingJump.objects.filter(user=user_with_records).order_by('-run_time')

        very_easy_count = TrainingJump.objects.filter(user=user_with_records, difficulty='VeryEasy').count()
        easy_count = TrainingJump.objects.filter(user=user_with_records, difficulty='Easy').count()
        normal_count = TrainingJump.objects.filter(user=user_with_records, difficulty='Normal').count()
        hard_count = TrainingJump.objects.filter(user=user_with_records, difficulty='Hard').count()
        very_hard_count = TrainingJump.objects.filter(user=user_with_records, difficulty='VeryHard').count()

        # 计算各项难度级别的平均训练时长
        very_easy_avg_time = TrainingJump.objects.filter(user=user_with_records, difficulty='VeryEasy').aggregate(avg_time=Avg('t_time'))['avg_time']
        very_easy_avg_time = round(very_easy_avg_time, 1) if very_easy_avg_time is not None else 0.0

        easy_avg_time = TrainingJump.objects.filter(user=user_with_records, difficulty='Easy').aggregate(avg_time=Avg('t_time'))['avg_time']
        easy_avg_time = round(easy_avg_time, 1) if easy_avg_time is not None else 0.0

        normal_avg_time = TrainingJump.objects.filter(user=user_with_records, difficulty='Normal').aggregate(avg_time=Avg('t_time'))['avg_time']
        normal_avg_time = round(normal_avg_time, 1) if normal_avg_time is not None else 0.0

        hard_avg_time = TrainingJump.objects.filter(user=user_with_records, difficulty='Hard').aggregate(avg_time=Avg('t_time'))['avg_time']
        hard_avg_time = round(hard_avg_time, 1) if hard_avg_time is not None else 0.0

        very_hard_avg_time = TrainingJump.objects.filter(user=user_with_records, difficulty='VeryHard').aggregate(avg_time=Avg('t_time'))['avg_time']
        very_hard_avg_time = round(very_hard_avg_time, 1) if very_hard_avg_time is not None else 0.0

        # 将记录数量数据放入上下文中，以便在HTML模板中使用
        context = {
            'user_name': user_name,
            'treatments': treatments,
            'VeryEasy': very_easy_count,
            'Easy': easy_count,
            'Normal': normal_count,
            'Hard': hard_count,
            'VeryHard': very_hard_count,
            'VeryEasy_time': very_easy_avg_time,
            'Easy_time': easy_avg_time,
            'Normal_time': normal_avg_time,
            'Hard_time': hard_avg_time,
            'VeryHard_time': very_hard_avg_time,
        }

        # 返回HTML页面渲染结果
        return render(request, 'chart_jump_ther.html', context)
    else:
        return HttpResponseForbidden("You are not authorized to access this page.")

@login_required
def chart_walk(request, user_id=None):
    # 获取当前登录用户
    current_user = request.user

    # 获取当前用户的步行训练数据，这里使用适当的过滤条件来获取用户自己的数据
    treatments = TrainingWalkStraight.objects.filter(user=current_user).order_by('-run_time')

    easy_count = TrainingWalkStraight.objects.filter(user=current_user, difficulty='Easy').count()
    normal_count = TrainingWalkStraight.objects.filter(user=current_user, difficulty='Normal').count()
    hard_count = TrainingWalkStraight.objects.filter(user=current_user, difficulty='Hard').count()

    # 计算各项难度级别的平均训练时长
    easy_avg_time = TrainingWalkStraight.objects.filter(user=current_user, difficulty='Easy').aggregate(avg_time=Avg('t_time'))['avg_time']
    easy_avg_time = round(easy_avg_time, 1) if easy_avg_time is not None else 0.0

    normal_avg_time = TrainingWalkStraight.objects.filter(user=current_user, difficulty='Normal').aggregate(avg_time=Avg('t_time'))['avg_time']
    normal_avg_time = round(normal_avg_time, 1) if normal_avg_time is not None else 0.0

    hard_avg_time = TrainingWalkStraight.objects.filter(user=current_user, difficulty='Hard').aggregate(avg_time=Avg('t_time'))['avg_time']
    hard_avg_time = round(hard_avg_time, 1) if hard_avg_time is not None else 0.0

    # 将记录数量数据放入上下文中，以便在HTML模板中使用
    context = {
        'treatments': treatments,
        'Easy': easy_count,
        'Normal': normal_count,
        'Hard': hard_count,
        'Easy_time': easy_avg_time,
        'Normal_time': normal_avg_time,
        'Hard_time': hard_avg_time,
    }

    # 返回HTML页面渲染结果
    return render(request, 'chart_walk.html', context)

@login_required
def chart_walk_ther(request, user_name):
    print("User name received:", user_name)  # 添加这行代码
    # 获取当前登录的治疗师用户
    user = request.user
    if user.user_type == 'therapist':
        # 使用 get_object_or_404 获取与用户名匹配的用户对象
        user_with_records = get_object_or_404(CustomUser, child_name=user_name)

        # 获取当前用户的跳舞训练数据，这里使用适当的过滤条件来获取用户自己的数据
        treatments = TrainingWalkStraight.objects.filter(user=user_with_records).order_by('-run_time')

        easy_count = TrainingWalkStraight.objects.filter(user=user_with_records, difficulty='Easy').count()
        normal_count = TrainingWalkStraight.objects.filter(user=user_with_records, difficulty='Normal').count()
        hard_count = TrainingWalkStraight.objects.filter(user=user_with_records, difficulty='Hard').count()

        easy_avg_time = TrainingWalkStraight.objects.filter(user=user_with_records, difficulty='Easy').aggregate(avg_time=Avg('t_time'))['avg_time']
        easy_avg_time = round(easy_avg_time, 1) if easy_avg_time is not None else 0.0

        normal_avg_time = TrainingWalkStraight.objects.filter(user=user_with_records, difficulty='Normal').aggregate(avg_time=Avg('t_time'))['avg_time']
        normal_avg_time = round(normal_avg_time, 1) if normal_avg_time is not None else 0.0

        hard_avg_time = TrainingWalkStraight.objects.filter(user=user_with_records, difficulty='Hard').aggregate(avg_time=Avg('t_time'))['avg_time']
        hard_avg_time = round(hard_avg_time, 1) if hard_avg_time is not None else 0.0
        # 将记录数量数据放入上下文中，以便在HTML模板中使用
        context = {
            'user_name': user_name,
            'treatments': treatments,
            'Easy': easy_count,
            'Normal': normal_count,
            'Hard': hard_count,
            'Easy_time': easy_avg_time,
            'Normal_time': normal_avg_time,
            'Hard_time': hard_avg_time,
        }

        # 返回HTML页面渲染结果
        return render(request, 'chart_walk_ther.html', context)
    else:
        return HttpResponseForbidden("You are not authorized to access this page.")

@login_required
def chart_onefoot(request, user_id=None):
    # 获取当前登录用户
    current_user = request.user

    # 获取当前用户的单脚训练数据，这里使用适当的过滤条件来获取用户自己的数据
    treatments = TrainingOneFoot.objects.filter(user=current_user).order_by('-run_time')

    very_easy_left_count = TrainingOneFoot.objects.filter(user=current_user, difficulty='VeryEasy_Left').count()
    very_easy_right_count = TrainingOneFoot.objects.filter(user=current_user, difficulty='VeryEasy_Right').count()
    easy_left_count = TrainingOneFoot.objects.filter(user=current_user, difficulty='Easy_Left').count()
    easy_right_count = TrainingOneFoot.objects.filter(user=current_user, difficulty='Easy_Right').count()
    normal_count = TrainingOneFoot.objects.filter(user=current_user, difficulty='Normal').count()
    hard_left_count = TrainingOneFoot.objects.filter(user=current_user, difficulty='Hard_Left').count()
    hard_right_count = TrainingOneFoot.objects.filter(user=current_user, difficulty='Hard_Right').count()

    # 计算各项难度级别的平均训练时长，并将结果转换为整数
    very_easy_left_avg_time = TrainingOneFoot.objects.filter(user=current_user, difficulty='VeryEasy_Left').aggregate(avg_time=Avg('t_time'))['avg_time']
    very_easy_left_avg_time = int(round(very_easy_left_avg_time, 0)) if very_easy_left_avg_time is not None else 0

    very_easy_right_avg_time = TrainingOneFoot.objects.filter(user=current_user, difficulty='VeryEasy_Right').aggregate(avg_time=Avg('t_time'))['avg_time']
    very_easy_right_avg_time = int(round(very_easy_right_avg_time, 0)) if very_easy_right_avg_time is not None else 0

    easy_left_avg_time = TrainingOneFoot.objects.filter(user=current_user, difficulty='Easy_Left').aggregate(avg_time=Avg('t_time'))['avg_time']
    easy_left_avg_time = int(round(easy_left_avg_time, 0)) if easy_left_avg_time is not None else 0

    easy_right_avg_time = TrainingOneFoot.objects.filter(user=current_user, difficulty='Easy_Right').aggregate(avg_time=Avg('t_time'))['avg_time']
    easy_right_avg_time = int(round(easy_right_avg_time, 0)) if easy_right_avg_time is not None else 0

    normal_avg_time = TrainingOneFoot.objects.filter(user=current_user, difficulty='Normal').aggregate(avg_time=Avg('t_time'))['avg_time']
    normal_avg_time = int(round(normal_avg_time, 0)) if normal_avg_time is not None else 0

    hard_left_avg_time = TrainingOneFoot.objects.filter(user=current_user, difficulty='Hard_Left').aggregate(avg_time=Avg('t_time'))['avg_time']
    hard_left_avg_time = int(round(hard_left_avg_time, 0)) if hard_left_avg_time is not None else 0

    hard_right_avg_time = TrainingOneFoot.objects.filter(user=current_user, difficulty='Hard_Right').aggregate(avg_time=Avg('t_time'))['avg_time']
    hard_right_avg_time = int(round(hard_right_avg_time, 0)) if hard_right_avg_time is not None else 0

    # 将记录数量数据放入上下文中，以便在HTML模板中使用
    context = {
        'treatments': treatments,
        'VeryEasy_Left_Count': very_easy_left_count,
        'VeryEasy_Right_Count': very_easy_right_count,
        'Easy_Left_Count': easy_left_count,
        'Easy_Right_Count': easy_right_count,
        'Normal_Count': normal_count,
        'Hard_Left_Count': hard_left_count,
        'Hard_Right_Count': hard_right_count,
        'VeryEasy_Left_time': very_easy_left_avg_time,
        'VeryEasy_Right_time': very_easy_right_avg_time,
        'Easy_Left_time': easy_left_avg_time,
        'Easy_Right_time': easy_right_avg_time,
        'Normal_time': normal_avg_time,
        'Hard_Left_time': hard_left_avg_time,
        'Hard_Right_time': hard_right_avg_time,
    }

    # 返回HTML页面渲染结果
    return render(request, 'chart_onefoot.html', context)

@login_required
def chart_onefoot_ther(request, user_name):
    print("User name received:", user_name)  # 添加这行代码
    # 获取当前登录的治疗师用户
    user = request.user
    if user.user_type == 'therapist':
        # 使用 get_object_or_404 获取与用户名匹配的用户对象
        user_with_records = get_object_or_404(CustomUser, child_name=user_name)

        # 获取当前用户的单脚训练数据，这里使用适当的过滤条件来获取用户自己的数据
        treatments = TrainingOneFoot.objects.filter(user=user_with_records).order_by('-run_time')

        very_easy_left_count = TrainingOneFoot.objects.filter(user=user_with_records, difficulty='VeryEasy_Left').count()
        very_easy_right_count = TrainingOneFoot.objects.filter(user=user_with_records, difficulty='VeryEasy_Right').count()
        easy_left_count = TrainingOneFoot.objects.filter(user=user_with_records, difficulty='Easy_Left').count()
        easy_right_count = TrainingOneFoot.objects.filter(user=user_with_records, difficulty='Easy_Right').count()
        normal_count = TrainingOneFoot.objects.filter(user=user_with_records, difficulty='Normal').count()
        hard_left_count = TrainingOneFoot.objects.filter(user=user_with_records, difficulty='Hard_Left').count()
        hard_right_count = TrainingOneFoot.objects.filter(user=user_with_records, difficulty='Hard_Right').count()

        # 计算各项难度级别的平均训练时长，并将结果转换为整数
        very_easy_left_avg_time = TrainingOneFoot.objects.filter(user=user_with_records, difficulty='VeryEasy_Left').aggregate(avg_time=Avg('t_time'))['avg_time']
        very_easy_left_avg_time = int(round(very_easy_left_avg_time, 0)) if very_easy_left_avg_time is not None else 0

        very_easy_right_avg_time = TrainingOneFoot.objects.filter(user=user_with_records, difficulty='VeryEasy_Right').aggregate(avg_time=Avg('t_time'))['avg_time']
        very_easy_right_avg_time = int(round(very_easy_right_avg_time, 0)) if very_easy_right_avg_time is not None else 0

        easy_left_avg_time = TrainingOneFoot.objects.filter(user=user_with_records, difficulty='Easy_Left').aggregate(avg_time=Avg('t_time'))['avg_time']
        easy_left_avg_time = int(round(easy_left_avg_time, 0)) if easy_left_avg_time is not None else 0

        easy_right_avg_time = TrainingOneFoot.objects.filter(user=user_with_records, difficulty='Easy_Right').aggregate(avg_time=Avg('t_time'))['avg_time']
        easy_right_avg_time = int(round(easy_right_avg_time, 0)) if easy_right_avg_time is not None else 0

        normal_avg_time = TrainingOneFoot.objects.filter(user=user_with_records, difficulty='Normal').aggregate(avg_time=Avg('t_time'))['avg_time']
        normal_avg_time = int(round(normal_avg_time, 0)) if normal_avg_time is not None else 0

        hard_left_avg_time = TrainingOneFoot.objects.filter(user=user_with_records, difficulty='Hard_Left').aggregate(avg_time=Avg('t_time'))['avg_time']
        hard_left_avg_time = int(round(hard_left_avg_time, 0)) if hard_left_avg_time is not None else 0

        hard_right_avg_time = TrainingOneFoot.objects.filter(user=user_with_records, difficulty='Hard_Right').aggregate(avg_time=Avg('t_time'))['avg_time']
        hard_right_avg_time = int(round(hard_right_avg_time, 0)) if hard_right_avg_time is not None else 0

        # 将记录数量数据放入上下文中，以便在HTML模板中使用
        context = {
            'user_name': user_name,
            'treatments': treatments,
            'VeryEasy_Left_Count': very_easy_left_count,
            'VeryEasy_Right_Count': very_easy_right_count,
            'Easy_Left_Count': easy_left_count,
            'Easy_Right_Count': easy_right_count,
            'Normal_Count': normal_count,
            'Hard_Left_Count': hard_left_count,
            'Hard_Right_Count': hard_right_count,
            'VeryEasy_Left_time': very_easy_left_avg_time,
            'VeryEasy_Right_time': very_easy_right_avg_time,
            'Easy_Left_time': easy_left_avg_time,
            'Easy_Right_time': easy_right_avg_time,
            'Normal_time': normal_avg_time,
            'Hard_Left_time': hard_left_avg_time,
            'Hard_Right_time': hard_right_avg_time,
        }

        # 返回HTML页面渲染结果
        return render(request, 'chart_onefoot_ther.html', context)
    else:
        return HttpResponseForbidden("You are not authorized to access this page.")

def yearly_data_api(request, year):
    try:
        # 獲取特定年份的舞蹈、單腳、跳高和直線走路的培訓數據
        dance_data = TrainingDance.objects.filter(run_time__year=year)
        onefoot_data = TrainingOneFoot.objects.filter(run_time__year=year)
        jump_data = TrainingJump.objects.filter(run_time__year=year)
        walk_data = TrainingWalkStraight.objects.filter(run_time__year=year)

        # 計算每個模式的數據總和
        dance_total = dance_data.aggregate(Sum('move_count'))['move_count__sum']
        onefoot_total = onefoot_data.aggregate(Sum('move_count'))['move_count__sum']
        jump_total = jump_data.aggregate(Sum('move_count'))['move_count__sum']
        walk_total = walk_data.aggregate(Sum('move_count'))['move_count__sum']

        # 構建 JSON 響應
        data = {
            'year': year,
            'data': {
                'dance': dance_total,
                'onefoot': onefoot_total,
                'jump': jump_total,
                'walk': walk_total,
            }
        }
        return JsonResponse(data)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)
def update_selection_status(request):
    if request.method == 'POST':
        image_id = request.POST.get('image_id')

        # 查找数据库记录，将选中状态更新为已选中
        try:
            image = User_BackgroundImage.objects.get(id=image_id)
            image.useornot = True
            image.save()
            return JsonResponse({'message': '更新成功'})
        except User_BackgroundImage.DoesNotExist:
            return JsonResponse({'message': '找不到指定的图像'}, status=404)
        except Exception as e:
            return JsonResponse({'message': '更新失败'}, status=500)

    # 处理其他请求方法或错误情况
    return JsonResponse({'message': '无效请求'}, status=400)

import cv2
from django.http import StreamingHttpResponse
from django.views.decorators import gzip
from . import jump123

# 开始摄像头
cap = cv2.VideoCapture(0)

@gzip.gzip_page
def video_feed(request):
    def generate():
        while True:
            success, image = cap.read()
            if not success:
                break

            # 使用mediapipe_detection.py中的函数处理图像
            annotated_image = jump123.jump_Per('VeryHard')

            # 将图像发送到前端
            _, buffer = cv2.imencode('.jpg', annotated_image)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

    return StreamingHttpResponse(generate(), content_type='multipart/x-mixed-replace; boundary=frame')

def video_display(request):
    return render(request, '123.html')