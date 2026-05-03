"""
dataset_loader.py  —  Semantic Search Engine for Kisan AI Chatbot v4
─────────────────────────────────────────────────────────────────────
Pipeline:
  User Query (Hinglish / Hindi)
    → Hinglish → Hindi token expansion
    → Crop detection
    → Intent detection
    → TF-IDF char-ngram cosine similarity
    → HARD THRESHOLD GATE: raw_sim >= 0.40 AND boosted >= 0.55
    → Returns empty list if nothing clears threshold (LLM-only mode)
"""

import os
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ─── Threshold: must pass BOTH raw cosine AND boosted score ───────────────────
SIMILARITY_THRESHOLD = 0.20   # raw TF-IDF cosine — lowered to catch more matches
SCORE_THRESHOLD      = 0.40   # after crop + intent boost

CROP_ALIASES = {
    "angoor": "अंगूर", "grape": "अंगूर", "grapes": "अंगूर",
    "anar": "अनार", "pomegranate": "अनार",
    "aam": "आम", "mango": "आम",
    "aalu": "आलू", "potato": "आलू", "aaloo": "आलू", "aloo": "आलू",
    "tamatar": "टमाटर", "tomato": "टमाटर",
    "pyaz": "प्याज", "onion": "प्याज",
    "gehun": "गेहूँ", "wheat": "गेहूँ", "gehu": "गेहूँ",
    "chawal": "चावल", "rice": "चावल", "dhan": "धान", "paddy": "धान",
    "makka": "मक्का", "maize": "मक्का", "corn": "मक्का", "makai": "मक्का",
    "sarso": "सरसों", "mustard": "सरसों", "sarson": "सरसों",
    "gobi": "गोभी", "gobhi": "गोभी", "cabbage": "गोभी", "cauliflower": "गोभी",
    "baingan": "बैंगन", "brinjal": "बैंगन", "eggplant": "बैंगन",
    "mirch": "मिर्च", "chilli": "मिर्च", "chili": "मिर्च",
    "ganna": "गन्ना", "sugarcane": "गन्ना", "gana": "गन्ना",
    "arhar": "अरहर", "toor": "अरहर",
    "moong": "मूंग", "mung": "मूंग",
    "adrak": "अदरक", "ginger": "अदरक",
    "haldi": "हल्दी", "turmeric": "हल्दी",
    "amrood": "अमरुद", "guava": "अमरुद",
    "kela": "केला", "banana": "केला",
    "papita": "पपीता", "papaya": "पपीता",
    "nimbu": "नींबू", "lemon": "नींबू",
    "anjeer": "अंजीर", "fig": "अंजीर",
    "cotton": "कपास", "kapas": "कपास",
    "soyabean": "सोयाबीन", "soya": "सोयाबीन", "soybean": "सोयाबीन",
    "chana": "चना", "chickpea": "चना",
    "masoor": "मसूर", "lentil": "मसूर",
    "amla": "आँवला", "gooseberry": "आँवला",
    "litchi": "लीची", "lychee": "लीची",
    "jamun": "जामुन",
    "nariyal": "नारियल", "coconut": "नारियल",
    "til": "तिल", "sesame": "तिल",
    "palak": "पालक", "spinach": "पालक",
    "lauki": "लौकी", "gourd": "लौकी",
    "kaddu": "कद्दू", "pumpkin": "कद्दू",
    "karela": "करेला", "bittergourd": "करेला",
    "tinda": "टिंडा", "torai": "तोरई",
    "shahtoot": "शहतूत",
    "peach": "आड़ू", "aadu": "आड़ू",
    "badam": "बादाम", "almond": "बादाम",
    "walnut": "अखरोट",
    "apple": "सेब", "seb": "सेब",
    "khira": "खीरा", "cucumber": "खीरा", "kheera": "खीरा",
    "bajra": "बाजरा", "millet": "बाजरा",
    "jowar": "ज्वार", "sorghum": "ज्वार",
}

HINGLISH_EXPAND = {
    "keede": "कीट कीड़े", "keet": "कीट", "kide": "कीड़े",
    "sundi": "सुंडी इल्ली", "illi": "इल्ली सुंडी",
    "makhi": "मक्खी", "aphid": "माहू चेपा",
    "mite": "माइट", "thrips": "थ्रिप्स",
    "whitefly": "सफेद मक्खी", "mealybug": "मिलीबग",
    "gandhi bug": "गंधी बग कीट",
    "gandhibug": "गंधी बग",
    "gandhi": "गंधी",
    "bug": "कीट बग",
    "grass hopper": "ग्रास हॉपर",
    "grasshopper": "ग्रास हॉपर टिड्डी",
    "tana bedhak": "तना बेधक",
    "stem borer": "तना बेधक",
    "patti lapetk": "पत्ती लपेटक",
    "leafroller": "पत्ती लपेटक",
    "khaira": "खैरा रोग",
    "leaf folder": "पत्ती लपेटक",
    "rice hispa": "हिस्पा",
    "bph": "भूरा फुदका",
    "armyworm": "सैनिक कीट इल्ली",
    "cutworm": "कटुआ इल्ली",
    "mealy bug": "मिलीबग",
    "scale insect": "शल्क कीट",
    "jassid": "जैसिड",
    "red spider": "लाल माइट",
    "rog": "रोग बीमारी", "bimari": "बीमारी रोग",
    "jhulsa": "झुलसा अंगमारी",
    "dhabbe": "धब्बे", "dabbe": "धब्बे",
    "sadan": "सड़न", "galn": "गलन",
    "fungal": "फफूंद", "fungus": "फफूंद कवक",
    "wilt": "मुरझाना उकठा", "uktha": "उकठा",
    "blast": "ब्लास्ट झोंका",
    "blight": "झुलसा",
    "mosaic": "मोज़ेक",
    "powdery mildew": "चूर्णिल आसिता",
    "canker": "कैंकर",
    "rust": "रतुआ",
    "tikka": "टिक्का रोग",
    "bakane": "बकानी रोग", "bakanai": "बकानी",
    "galan": "गलन सड़न",
    "patte": "पत्ते पत्तियाँ", "pattiya": "पत्तियाँ पत्ते",
    "pedh": "पेड़ पौधा", "podha": "पौधा",
    "jad": "जड़", "tana": "तना",
    "phool": "फूल", "fal": "फल", "beej": "बीज",
    "chhaal": "छाल", "shakh": "शाखा",
    "pila": "पीला", "peela": "पीला पीली",
    "sukh": "सूखना सूखा", "sukha": "सूखना",
    "nahi lag": "नहीं लगना",
    "gir": "गिरना गिर",
    "murjha": "मुरझाना",
    "kaale": "काले धब्बे", "kala": "काला",
    "safed": "सफेद", "lal": "लाल",
    "paani": "पानी", "pani": "पानी सिंचाई",
    "sinchai": "सिंचाई पानी",
    "drip": "ड्रिप सिंचाई",
    "kitna": "कितना", "kab": "कब",
    "kitne din": "कितने दिन",
    "lagaye": "लगाएं दें",
    "chahiye": "चाहिए",
    "dena chahiye": "देना चाहिए",
    "khad": "खाद उर्वरक", "urvarak": "उर्वरक खाद",
    "dap": "DAP डीएपी", "urea": "यूरिया",
    "kami": "कमी", "zinc": "जिंक", "potash": "पोटाश",
    "kha rahe": "खा रहे नुकसान",
    "kha raha": "खा रहा",
    "lag gaye": "लगना आक्रमण",
    "kaise kare": "कैसे करें",
    "kab dale": "कब डालें",
    "madad": "मदद समाधान",
    "kripya": "कृपया",
    "batao": "बताएं",
    "upay": "उपाय समाधान",
    "rokne": "रोकना नियंत्रण",
    "bachao": "बचाव",
}

INTENT_KEYWORDS = {
    "pest": [
        "कीड़", "कीट", "इल्ली", "सुंडी", "मक्खी", "माहू", "चेपा",
        "aphid", "keede", "keet", "kide", "sundi",
        "whitefly", "thrips", "mite", "माइट", "mealybug", "bug",
        "कृमि", "लट", "मिलीबग", "खा रहे", "kha rahe",
        "gandhi", "grasshopper", "stem borer", "tana bedhak",
        "hispa", "armyworm",
    ],
    "disease": [
        "रोग", "बीमारी", "फफूंद", "fungal", "blight", "blast", "spot",
        "झुलसा", "सड़न", "गलन", "rog", "bimari", "dhabbe", "dabbe",
        "canker", "wilt", "mosaic", "धब्बे", "काले", "मुरझा",
        "bakane", "khaira", "tikka", "rust", "रतुआ",
    ],
    "fertilizer": [
        "खाद", "उर्वरक", "NPK", "DAP", "यूरिया", "पोषण", "khad",
        "fertilizer", "deficiency", "कमी", "पीली", "yellow",
        "नाइट्रोजन", "फास्फोरस", "पोटाश", "kami", "urea", "zinc",
    ],
    "irrigation": [
        "पानी", "सिंचाई", "irrigation", "water", "paani", "sinchai", "pani",
        "नमी", "moisture", "drought", "सूखा", "drip", "ड्रिप", "kitna",
        "कितना", "कब दें", "कब लगाएं", "दिनों बाद",
    ],
    "growth": [
        "विकास", "growth", "बढ़वार", "फल", "flower", "फूल", "yield",
        "उपज", "production", "नहीं लग", "गिर", "खेती", "बुवाई",
        "sowing", "harvesting", "कटाई",
    ],
    "cultivation": [
        "ugane", "ugana", "ugaye", "kaise ugaye", "kaise ugayen",
        "kaise kare", "kaise karen", "kaise lagaye", "kaise lagayen",
        "kya chahiye", "kya kya chahiye", "chahiye hoga",
        "kheti kaise", "kheti karna", "kheti ke liye",
        "ugaane", "ugaana",
        "उगाना", "उगाएं", "उगाने", "खेती कैसे", "खेती करना",
        "कैसे उगाएं", "क्या चाहिए", "लगाने का तरीका",
        "बुवाई कैसे", "रोपाई", "बीज कैसे", "खेत तैयार",
        "how to grow", "how to cultivate", "cultivation",
    ],
    "general": [
        "कैसे", "how", "जानकारी", "information", "advice", "सलाह",
        "kaise", "batao", "bataiye",
    ],
}

INTENT_LABELS = {
    "pest":        "कीट / पतंगे की समस्या",
    "disease":     "रोग / बीमारी की समस्या",
    "fertilizer":  "पोषण / उर्वरक की समस्या",
    "irrigation":  "पानी / सिंचाई की समस्या",
    "growth":      "विकास / उपज की समस्या",
    "cultivation": "खेती / उगाने की जानकारी",
    "general":     "सामान्य जानकारी",
}


class KisanDataset:
    def __init__(self, filepath="adv_data.xlsx"):
        print("📂 Loading dataset...", flush=True)
        df_raw = pd.read_excel(filepath)
        self.df = (
            df_raw[["problem", "solution", "cropname"]]
            .dropna(subset=["cropname", "problem"])
            .fillna("")
            .copy()
        )
        self.df["cropname"] = self.df["cropname"].astype(str).str.strip()
        self.df["problem"]  = self.df["problem"].astype(str).str.strip()
        self.df["solution"] = self.df["solution"].astype(str).str.strip()
        self.df = self.df[self.df["problem"].str.len() > 5].reset_index(drop=True)
        self.crop_list = self.df["cropname"].unique().tolist()

        self.df["_index_text"] = self.df["cropname"] + " " + self.df["problem"]
        self._vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 5),
            min_df=1,
            max_features=50000,
            sublinear_tf=True,
        )
        self._matrix = self._vectorizer.fit_transform(self.df["_index_text"])
        print(f"✅ Index built: {len(self.df)} rows, {self._matrix.shape[1]} features", flush=True)

    def _expand_query(self, query: str) -> str:
        expanded = query
        q_lower  = query.lower()
        for roman in sorted(HINGLISH_EXPAND.keys(), key=len, reverse=True):
            if roman in q_lower:
                expanded += " " + HINGLISH_EXPAND[roman]
        for alias, hindi_crop in CROP_ALIASES.items():
            if alias in q_lower:
                expanded += " " + hindi_crop
        return expanded

    # Non-crop words that can appear as cropname in dirty dataset rows
    _NON_CROP = {"पानी", "खेत", "खेती", "मिट्टी", "बीज", "पत्ती", "पत्ते", "जड़"}

    # Words that contain crop names as substrings — must not false-match
    _CROP_BLOCKLIST = {"समाधान", "विधान", "प्रधान", "निधान", "साधान", "आधान"}

    def detect_crop(self, query: str):
        q_lower = query.lower()
        # Longest alias first — avoids "aam" matching inside "naam"
        for alias in sorted(CROP_ALIASES.keys(), key=len, reverse=True):
            if re.search(r'\b' + re.escape(alias) + r'\b', q_lower):
                return CROP_ALIASES[alias], alias
        # Direct Hindi crop name — match only as whole token (space/boundary bounded)
        # Split query into tokens to avoid "धान" matching inside "समाधान"
        tokens = set(re.split(r'[\s,।!?]+', query))
        # Also check "cropके", "cropमें" etc. by checking if any token starts with crop
        for crop in sorted(self.crop_list, key=len, reverse=True):
            if len(crop) < 2 or crop in self._NON_CROP:
                continue
            for token in tokens:
                if token == crop or token.startswith(crop + 'के') or \
                   token.startswith(crop + 'में') or token.startswith(crop + 'पर') or \
                   token.startswith(crop + 'की') or token.startswith(crop + 'का'):
                    return crop, crop
        return None, None

    def detect_intent(self, query: str) -> str:
        q_lower = query.lower()
        scores = {k: 0 for k in INTENT_KEYWORDS}
        for intent, keywords in INTENT_KEYWORDS.items():
            for kw in keywords:
                if kw.lower() in q_lower:
                    scores[intent] += 1
        best = max(scores, key=scores.get)
        return best if scores[best] > 0 else "general"

    def retrieve(self, query: str, crop_hindi=None, intent="general", top_k=5) -> list:
        """
        Returns matched rows only if raw cosine >= SIMILARITY_THRESHOLD
        AND boosted score >= SCORE_THRESHOLD.
        Returns [] when no good match — caller switches to LLM-only mode.
        """
        expanded = self._expand_query(query)
        q_vec    = self._vectorizer.transform([expanded])
        raw_sim  = cosine_similarity(q_vec, self._matrix)[0]
        boosted  = raw_sim.copy()

        if crop_hindi:
            exact = self.df["cropname"] == crop_hindi
            boosted[exact] += 0.45
            if not exact.any():
                partial = self.df["cropname"].str.startswith(crop_hindi[:2], na=False)
                boosted[partial] += 0.20

        intent_kws = [k.lower() for k in INTENT_KEYWORDS.get(intent, [])]
        prob_lower = self.df["problem"].str.lower()
        for kw in intent_kws:
            boosted[prob_lower.str.contains(re.escape(kw), na=False)] += 0.05

        # ── HARD THRESHOLD GATE ───────────────────────────────────────────────
        valid = (raw_sim >= SIMILARITY_THRESHOLD) & (boosted >= SCORE_THRESHOLD)
        # Crop isolation: if user specified a crop, ONLY return rows for that crop.
        # Prevents खीरे/गेहूँ rows leaking into धान queries.
        if crop_hindi:
            valid = valid & (self.df["cropname"] == crop_hindi)
        if not valid.any():
            print(f"⚠  Threshold not met (best raw={raw_sim.max():.3f}) → LLM-only", flush=True)
            return []

        valid_idx = np.where(valid)[0]
        ranked    = valid_idx[np.argsort(boosted[valid_idx])[::-1]][:top_k]

        results = []
        for idx in ranked:
            row = self.df.iloc[idx]
            if not row["problem"] or not row["solution"]:
                continue
            results.append({
                "problem":    row["problem"],
                "solution":   row["solution"],
                "crop":       row["cropname"],
                "similarity": float(raw_sim[idx]),
                "score":      float(boosted[idx]),
            })
        return results


_dataset = None

def get_dataset() -> KisanDataset:
    global _dataset
    if _dataset is None:
        dataset_path = os.path.join(os.path.dirname(__file__), "adv_data.xlsx")
        _dataset = KisanDataset(dataset_path)
    return _dataset
