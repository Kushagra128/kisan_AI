"""
unanswered_problems_logger.py
-----------------------------------
Backend helper to log unanswered / new problems from Kisan Mitra
into PostgreSQL.

Call save_problem() from your Flask/Django route when a new
query arrives that is NOT in the database.

Usage:
    from unanswered_problems_logger import save_problem
    save_problem(query="Meri fasal pe daag aa rahe hain", brief_solution="Fungal - Mancozeb spray")
"""

import os
from datetime import datetime
from chatbot.models import UnansweredProblem

CATEGORY_MAP = {
    'rog': 'Fasal Rog', 'disease': 'Fasal Rog', 'fungal': 'Fasal Rog', 'dhabb': 'Fasal Rog',
    'keet': 'Keet Niyantran', 'insect': 'Keet Niyantran', 'pest': 'Keet Niyantran',
    'khad': 'Poshan Prabandhan', 'fertil': 'Poshan Prabandhan', 'urea': 'Poshan Prabandhan', 'npk': 'Poshan Prabandhan',
    'sinch': 'Jal Prabandhan', 'water': 'Jal Prabandhan', 'irrigation': 'Jal Prabandhan',
    'mausam': 'Mausam', 'weather': 'Mausam',
}

def detect_category(query: str) -> str:
    q = query.lower()
    for keyword, category in CATEGORY_MAP.items():
        if keyword in q:
            return category
    return 'Anya'


def save_problem(query: str, brief_solution: str, category: str = None, status: str = 'Pending Review') -> int:
    """
    Append a new unanswered problem row to PostgreSQL database.

    Args:
        query           : The user's original question text
        brief_solution  : One-line solution from adv_data.xlsx or fallback
        category        : Auto-detected if None
        status          : Default 'Pending Review'

    Returns:
        The ID of the created or updated UnansweredProblem record.
    """
    if category is None:
        category = detect_category(query)

    try:
        # Search if query already exists
        problem_record = UnansweredProblem.objects.filter(query__iexact=query.strip()).first()

        if problem_record:
            # Update existing row
            problem_record.timestamp = datetime.now()
            # If we wanted to update brief_solution, we'd need a field for it, 
            # currently the model only has query, detected_intent, detected_crop, timestamp.
            problem_record.detected_intent = category
            problem_record.save()
            print(f"[unanswered_logger] Updated record {problem_record.id} for query: {query[:60]}...")
            return problem_record.id
        else:
            # Create new record
            problem_record = UnansweredProblem.objects.create(
                query=query.strip(),
                detected_intent=category,
            )
            print(f"[unanswered_logger] Saved new record {problem_record.id}: {query[:60]}...")
            return problem_record.id
    except Exception as e:
        print(f"[unanswered_logger] Error saving to DB: {e}")
        return -1


# -------------------------------------------------------
# CLI test
# -------------------------------------------------------
if __name__ == '__main__':
    import django
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'kisan_project.settings')
    django.setup()
    
    print("Testing logger...")
    save_problem(
        query="Meri tamatar ki fasal mein patta mur raha hai",
        brief_solution="Viral disease - copper fungicide + aphid control karein",
        category="Fasal Rog"
    )
    save_problem(
        query="Sarson mein aphid ka attack ho gaya hai",
        brief_solution="Imidacloprid 0.5ml/L water mein spray karein",
    )
    print("Done.")