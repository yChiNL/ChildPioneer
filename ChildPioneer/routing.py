from django.urls import path
from . import consumers

websocket_urlpatterns = [
    path('ws/detect_dance/', consumers.DetectionConsumer.as_asgi()),

]