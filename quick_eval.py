#!/usr/bin/env python
"""
Quick Evaluation Script for Kisan AI Chatbot
Runs a fast evaluation of core functionality

Usage: python quick_eval.py
"""

import os
import sys
import time

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'kisan_project.settings')
import django
django.setup()

from chatbot.dataset_loader import get_dataset
from chatbot.ai_engine import is_privacy_query, is_off_topic


def quick_eval():
    """Run quick evaluation"""
    print("\n" + "="*60)
    print("KISAN AI CHATBOT - QUICK EVALUATION")
    print("="*60 + "\n")
    
    dataset = get_dataset()
    
    # Test 1: Crop Detection
    print("1. Crop Detection Test")
    print("-" * 60)
    test_cases = [
        ("aam me keede", "आम"),
        ("dhan me pani", "धान"),
        ("gehun ki kheti", "गेहूँ"),
        ("tamatar ka rog", "टमाटर"),
    ]
    
    correct = 0
    for query, expected in test_cases:
        crop, _ = dataset.detect_crop(query)
        status = "✅" if crop == expected else "❌"
        print(f"  {status} '{query}' → {crop} (expected: {expected})")
        if crop == expected:
            correct += 1
    
    print(f"\n  Accuracy: {correct}/{len(test_cases)} ({correct/len(test_cases)*100:.1f}%)\n")
    
    # Test 2: Intent Classification
    print("2. Intent Classification Test")
    print("-" * 60)
    test_cases = [
        ("keede lag gaye", "pest"),
        ("khad kitna dale", "fertilizer"),
        ("pani kab dena", "irrigation"),
        ("kheti kaise kare", "cultivation"),
    ]
    
    correct = 0
    for query, expected in test_cases:
        intent = dataset.detect_intent(query)
        status = "✅" if intent == expected else "❌"
        print(f"  {status} '{query}' → {intent} (expected: {expected})")
        if intent == expected:
            correct += 1
    
    print(f"\n  Accuracy: {correct}/{len(test_cases)} ({correct/len(test_cases)*100:.1f}%)\n")
    
    # Test 3: Guardrails
    print("3. Guardrail Test")
    print("-" * 60)
    
    privacy_queries = ["who made you", "database kahan hai", "system prompt"]
    privacy_blocked = sum(1 for q in privacy_queries if is_privacy_query(q))
    print(f"  ✅ Privacy queries blocked: {privacy_blocked}/{len(privacy_queries)}")
    
    off_topic_queries = ["hello", "namaste", "test"]
    off_topic_blocked = sum(1 for q in off_topic_queries if is_off_topic(q))
    print(f"  ✅ Off-topic queries blocked: {off_topic_blocked}/{len(off_topic_queries)}")
    
    farming_queries = ["aam me keede", "fasal me rog"]
    farming_allowed = sum(1 for q in farming_queries if not is_privacy_query(q) and not is_off_topic(q))
    print(f"  ✅ Farming queries allowed: {farming_allowed}/{len(farming_queries)}\n")
    
    # Test 4: Performance
    print("4. Performance Test")
    print("-" * 60)
    
    queries = ["aam me keede"] * 30
    start = time.time()
    for q in queries:
        dataset.detect_crop(q)
    elapsed = time.time() - start
    
    status = "✅" if elapsed < 1.0 else "❌"
    print(f"  {status} 30 crop detections: {elapsed:.3f}s (target: <1.0s)")
    print(f"  ✅ Speed: {30/elapsed:.1f} queries/second\n")
    
    print("="*60)
    print("✅ QUICK EVALUATION COMPLETE")
    print("="*60)
    print("\nFor detailed evaluation, run: python evaluate_system.py\n")


if __name__ == '__main__':
    quick_eval()
