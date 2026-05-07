from django.shortcuts import render
from django.http import JsonResponse, StreamingHttpResponse
import json

from .ai_engine import stream_answer, get_answer
from .audio_engine import transcribe_wav, synthesize_wav
from django.views.decorators.csrf import csrf_exempt

def index(request):
    return render(request, "landing_gov.html")

def chat_page(request):
    return render(request, "index.html")

def widget(request):
    return render(request, "widget.html")

@csrf_exempt
def chat_stream(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
        except json.JSONDecodeError:
            data = {}
        
        # Validate message field exists and is a string
        message = data.get("message", "")
        if not isinstance(message, str):
            return JsonResponse({"error": "संदेश एक स्ट्रिंग होना चाहिए।"}, status=400)
            
        if not data or not message.strip():
            return JsonResponse({"error": "संदेश खाली है।"}, status=400)

        user_message = message.strip()
        
        # Generator for StreamingHttpResponse
        def generate():
            for chunk in stream_answer(user_message):
                yield chunk
                
        response = StreamingHttpResponse(
            generate(),
            content_type="text/event-stream"
        )
        response["Cache-Control"] = "no-cache"
        response["X-Accel-Buffering"] = "no"
        return response
    return JsonResponse({"error": "Method not allowed"}, status=405)

@csrf_exempt
def chat(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
        except json.JSONDecodeError:
            data = {}
        
        # Validate message field exists and is a string
        message = data.get("message", "")
        if not isinstance(message, str):
            return JsonResponse({"error": "संदेश एक स्ट्रिंग होना चाहिए।"}, status=400)
            
        if not data or not message.strip():
            return JsonResponse({"error": "संदेश खाली है।"}, status=400)

        user_message = message.strip()
        result = get_answer(user_message)

        return JsonResponse({
            "response":       result.get("response"),
            "crop":           result.get("crop"),
            "intent":         result.get("intent"),
            "source":         result.get("source"),
            "suggestions":    result.get("suggestions", []),
            "top_similarity": result.get("top_similarity", 0),
        })
    return JsonResponse({"error": "Method not allowed"}, status=405)

def initial_suggestions(request):
    return JsonResponse({"suggestions": [
        "टमाटर में पत्ते पीले हो रहे हैं",
        "गेहूँ में कीड़े लग गए हैं",
        "आलू में खाद कब डालें",
        "धान में पानी कितना देना चाहिए",
    ]})

def health(request):
    return JsonResponse({"status": "ok", "model": "llama3.2", "streaming": True})

@csrf_exempt
def stt_view(request):
    if request.method == "POST":
        audio_file = request.FILES.get("audio")
        if not audio_file:
            return JsonResponse({"error": "No audio file provided"}, status=400)
        
        audio_data = audio_file.read()
        text = transcribe_wav(audio_data)
        return JsonResponse({"text": text})
    return JsonResponse({"error": "Method not allowed"}, status=405)

@csrf_exempt
def tts_view(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            text = data.get("text", "")
            if not text:
                return JsonResponse({"error": "No text provided"}, status=400)
            
            filename = synthesize_wav(text)
            if filename:
                return JsonResponse({"url": f"/static/audio/{filename}"})
            else:
                return JsonResponse({"error": "TTS failed"}, status=500)
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)
    return JsonResponse({"error": "Method not allowed"}, status=405)
