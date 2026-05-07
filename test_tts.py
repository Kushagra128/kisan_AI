import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'kisan_project.settings')
django.setup()

from chatbot.audio_engine import synthesize_wav

print("Testing TTS...")
filename = synthesize_wav("नमस्ते किसान भाई, मैं आपकी क्या मदद कर सकता हूँ?")
if filename:
    print(f"Success! Generated file: {filename}")
else:
    print("TTS failed.")
