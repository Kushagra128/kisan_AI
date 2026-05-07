import pyttsx3
import pythoncom
pythoncom.CoInitialize()
engine = pyttsx3.init()
voices = engine.getProperty('voices')
for v in voices:
    print(f"Name: {v.name}")
    print(f"ID: {v.id}")
    print(f"Languages: {v.languages}")
    print("---")
