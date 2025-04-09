from django.contrib import admin
from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from pythonProject3 import views
from pythonProject3 import consumers
from . import routing
from .views import *
from django.contrib.auth import views as auth_views





urlpatterns = [
    path('admin/', admin.site.urls),
    path('Training_mode', views.Training_mode, name="Training_mode"),
    path('Therapist_training/', views.therapist_training_view, name='Therapist_training'),
    path('Data/', views.dataimport, name='dataimport'),
    path('buttontest/', views.buttontest, name='buttontest'),

    path('resetPwd', views.forgot_password, name='forgot_password'),

    path('', views.user_login, name='login'),
    path('logout/', views.user_logout, name='logout'),
    path('identitySelect/', views.identity, name='identitySelect'),
    path('identitySelect/student_register/', views.register_user, name='student_register'),
    path('identitySelect/therapist_register/', views.register_therapist, name='therapist_register'),
    path('student_reviseprofile/', views.student_reviseprofile, name='student_reviseprofile'),

    path('update_profile/', views.update_profile, name='update_profile'),
    path('change_password/', views.change_password, name='change_password'),
    path('change_password_ther/', views.change_password, name='change_password_ther'),
    path('password_change_done/', views.password_change_done, name='password_change_done'),
    path('password_change_done_ther/', views.password_change_done_ther, name='password_change_done_ther'),
    path('therapist_update_profile/', views.update_profile, name='therapist_update_profile'),

    path('add_friend/', views.add_friend, name='add_friend'),
    path('delete_friend/<int:friend_id>/', views.delete_friend, name='delete_friend'),
    path('therapist_home/', views.therapist_home, name='therapist_home'),
    path('user_home/', views.user_home, name='user_home'),

    path('myprofile/', views.myprofile, name='myprofile'),


    path('media', views.media, name='resetpasssword'),
    path('contacttherapist/', views.contact_therapists, name='contact_therapists'),
    path('contact_user/<int:user_id>/', views.contact_user, name='contact_user'),

    #治療師規劃，需加入id
    path('Therapist_training/dance/<str:difficulty>/<str:frequency>/<str:id>/', views.dance_view, name='dance'),
    path('Therapist_training/onefoot/<str:difficulty>/<str:frequency>/<str:id>/', views.onefoot_therapist_view, name='onefoot'),
    path('Therapist_training/jump/<str:difficulty>/<str:frequency>/<str:id>/', views.jump_therapist_view, name='jump'),
    path('Therapist_training/walkstraight/<str:difficulty>/<str:frequency>/<str:id>/', views.walkstraight_therapist_view, name='walkstraight'), #要改

    path('Personal_TrainSelect/', views.Personal_TrainSelect, name='Personal_TrainSelect'),
    path('Personal_training/', views.Personal_training, name='Personal_training'),
    path('Personal_training_jump/', views.Personal_training_jump, name='Personal_training_jump'),
    path('Personal_training_onefoot/', views.Personal_training_onefoot, name='Personal_training_onefoot'),
    path('Personal_training_walk/', views.Personal_training_walk, name='Personal_training_walk'),
    path('chart_index.html', views.chart_index, name='chart_index'),
    path('chart_index_ther/<str:user_name>/', views.chart_index_ther, name='chart_index_ther'),
    path('schedule_treatment/<int:user_id>/', views.schedule_treatment, name='schedule_treatment'),
    path('dance_select_mode/', views.dance_SelectMode, name='select_mode'),
    path('onefoot_select_mode/', views.dance_SelectMode, name='select_mode'),
    path('jump_select_mode/', views.jump_SelectMode, name='select_mode'),
    path('start/<str:mode>/', views.dance_start, name='dance_start'),
    path('dance_start/<str:mode>/', views.dance_start, name='dance_start'),
    path('onefoot_start/<str:mode>/', views.onefoot_start, name='onefoot_start'),
    path('jump_start/<str:mode>/', views.jump_start, name='jump_start'),
    path('walk_start/<str:mode>/', views.walk_start, name='walk_start'),
    path('collections/', views.collections, name='collections'),
    path('update_selection_status/', views.update_selection_status, name='update_selection_status'),

    path('store/', views.store, name='store'),
    path('purchase_background/<int:background_id>/', views.purchase_background, name='purchase_background'),

    path('api/training-data/', views.get_training_data, name='get_training_data'),
    path('get_training_data_year_ther/<str:user_name>/', views.get_training_data_year_ther,name='get_training_data_year_ther'),
    path('get_training_data_month_ther/<str:user_name>/', views.get_training_data_month_ther,name='get_training_data_month_ther'),
    path('get_training_data_week_ther/<str:user_name>/', views.get_training_data_week_ther,name='get_training_data_week_ther'),
    path('get_training_data_day_ther/<str:user_name>/', views.get_training_data_day_ther,name='get_training_data_day_ther'),
    path('api/dance-training-data/', views.get_dance_training_data, name='get_dance_training_data'),
    path('api/dance-training-data-ther/<str:user_name>/<str:date>/', views.get_dance_training_data_ther,name='get_dance_training_data_ther'),
    path('api/jump_training_data/', views.get_jump_training_data, name='get_jump_training_data'),
    path('api/jump-training-data-ther/<str:user_name>/<str:date>/', views.get_jump_training_data_ther,name='get_jump_training_data_ther'),
    path('api/onefoot_training_data/', views.get_onefoot_training_data, name='get_onefoot_training_data'),
    path('api/onefoot-training-data-ther/<str:user_name>/<str:date>/', views.get_onefoot_training_data_ther,name='get_onefoot_training_data_ther'),
    path('api/walk_training_data/', views.get_walk_training_data, name='get_walk_training_data'),
    path('api/walk-training-data-ther/<str:user_name>/<str:date>/', views.get_walk_training_data_ther,name='get_walk_training_data_ther'),
    path('api/training-data-year/', views.get_training_data_year, name='get_training_data_year'),
    path('api/training-data-month/', views.get_training_data_month, name='get_training_data_month'),
    path('api/training-data-week/', views.get_training_data_week, name='get_training_data_week'),
    path('api/training-data-day/', views.get_training_data_day, name='get_training_data_day'),
    path('chart_dance.html', views.chart_dance, name='chart_dance'),
    path('chart_dance_ther/<str:user_name>/', views.chart_dance_ther, name='chart_dance_ther'),
    path('chart_jump.html', views.chart_jump, name='chart_jump'),
    path('chart_jump_ther/<str:user_name>/', views.chart_jump_ther, name='chart_jump_ther'),
    path('chart_walk.html', views.chart_walk, name='chart_walk'),
    path('chart_walk_ther/<str:user_name>/', views.chart_walk_ther, name='chart_walk_ther'),
    path('chart_onefoot.html', views.chart_onefoot, name='chart_onefoot'),
    path('chart_onefoot_ther/<str:user_name>/', views.chart_onefoot_ther, name='chart_onefoot_ther'),

    #subchart
    path('api/training-data-dance/<str:date>/', views.get_training_data_dance, name='get_training_data_dance'),
    path('api/training-data-jump/<str:date>/', views.get_training_data_jump, name='get_training_data_jump'),
    path('api/training-data-walk/<str:date>/', views.get_training_data_walk, name='get_training_data_walk'),
    path('api/training-data-foot/<str:date>/', views.get_training_data_foot, name='get_training_data_foot'),

    path('api/yearly-data/<int:year>/', views.yearly_data_api, name='yearly_data_api'),


    path('video_feed/', views.video_feed, name='video_feed'),
    path('video_display/', views.video_display, name='123'),

    path('walkstraight/', views.walkstraight_view, name='walkstraight'),
    path('delete_friend/<int:friend_id>/', views.delete_friend, name='delete_friend'),

]

websocket_urlpatterns = routing.websocket_urlpatterns

websocket_urlpatterns = [
    path('ws/detect_dance/', consumers.DetectionConsumer.as_asgi()),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)