from django.db import models
from django.utils import timezone

class AgriculturalAdvice(models.Model):
    cropname = models.CharField(max_length=255)
    problem = models.TextField()
    solution = models.TextField()

    def __str__(self):
        return f"{self.cropname} - {self.problem[:50]}"

class UnansweredProblem(models.Model):
    query = models.TextField()
    detected_intent = models.CharField(max_length=100)
    detected_crop = models.CharField(max_length=100, null=True, blank=True)
    timestamp = models.DateTimeField(default=timezone.now)

    def __str__(self):
        return f"[{self.timestamp}] {self.query[:50]}"
