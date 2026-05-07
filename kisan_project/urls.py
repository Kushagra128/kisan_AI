from django.contrib import admin
from django.urls import path
from chatbot import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.index, name='index'),
    path('chat-app/', views.chat_page, name='chat_page'),
    path('widget', views.widget, name='widget'),
    path('chat/stream', views.chat_stream, name='chat_stream'),
    path('chat', views.chat, name='chat'),
    path('chat/stt', views.stt_view, name='stt_view'),
    path('chat/tts', views.tts_view, name='tts_view'),
    path('initial-suggestions', views.initial_suggestions, name='initial_suggestions'),
    path('health', views.health, name='health'),
]
