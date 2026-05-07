import os
import io
import wave
import json
import uuid
from django.conf import settings

MODEL_DIR = os.path.join(settings.BASE_DIR, "models", "vosk-model-hi-0.22")
_stt_model = None

def get_stt_model():
    global _stt_model
    if _stt_model is None:
        if os.path.exists(MODEL_DIR):
            # Lazy import vosk only when actually needed
            import vosk
            vosk.SetLogLevel(-1)
            _stt_model = vosk.Model(MODEL_DIR)
        else:
            print(f"STT Model not found at {MODEL_DIR}")
    return _stt_model

def transcribe_wav(audio_data: bytes) -> str:
    """Expects a WAV file with header (16kHz, mono, 16-bit PCM)."""
    from vosk import KaldiRecognizer
    model = get_stt_model()
    if not model:
        return ""
    
    try:
        with io.BytesIO(audio_data) as f:
            with wave.open(f, "rb") as wf:
                if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() not in [16000, 8000]:
                    print("Invalid audio format for Vosk. Must be 16kHz mono 16-bit.")
                    return ""
                
                rec = KaldiRecognizer(model, wf.getframerate())
                rec.SetWords(False)
                
                frames = wf.readframes(wf.getnframes())
                rec.AcceptWaveform(frames)
                res = json.loads(rec.FinalResult())
                return res.get("text", "")
    except Exception as e:
        print(f"Error in STT: {e}")
        return ""

def synthesize_wav(text: str) -> str:
    """Returns filename of generated wav. Imports are lazy to avoid COM hang on startup."""
    try:
        # Lazy imports — keep pyttsx3 & pythoncom out of module-level to avoid
        # blocking Django's system check on startup (COM init can hang the process)
        import pythoncom
        import pyttsx3

        pythoncom.CoInitialize()
        engine = pyttsx3.init()

        # Find Hindi voice (Microsoft Kalpana or Hemant)
        voices = engine.getProperty('voices')
        hi_voice = None
        for v in voices:
            if 'Hindi' in v.name or 'hi-IN' in v.id or 'Hemant' in v.name or 'Kalpana' in v.name:
                hi_voice = v.id
                break
        if hi_voice:
            engine.setProperty('voice', hi_voice)
        
        engine.setProperty('rate', 150)
        
        filename = f"{uuid.uuid4().hex}.wav"
        filepath = os.path.join(settings.BASE_DIR, "static", "audio", filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        engine.save_to_file(text, filepath)
        engine.runAndWait()
        del engine
        return filename
    except Exception as e:
        print(f"Error in TTS: {e}")
        return ""
    finally:
        try:
            import pythoncom
            pythoncom.CoUninitialize()
        except Exception:
            pass
