"""
ai_engine.py  —  Kisan AI Engine v4
─────────────────────────────────────
Pipeline:
  Query → Detect Crop+Intent → Retrieve (threshold-gated)
        → If matches found: Dataset-augmented prompt
        → If no matches:    LLM-only expert prompt
        → Ollama streaming → yield chunks
        → Fallback: structured dataset answer
"""

import json
import requests
from .dataset_loader import get_dataset, INTENT_KEYWORDS, INTENT_LABELS, CROP_ALIASES
from .unanswered_problems_logger import save_problem
import re

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL      = "llama3.2"

import threading

def async_save_problem(query, brief_solution, category):
    threading.Thread(
        target=save_problem,
        kwargs={"query": query, "brief_solution": brief_solution, "category": category},
        daemon=True
    ).start()

def extract_brief_solution(text: str) -> str:
    """Extract a brief 1-line solution from LLM generated structured text."""
    match = re.search(r"(?:समाधान|उपाय).*?[:\n](.*?)(?:\n\n|\Z|\n\s*(?:⚠|सावधानियां|अतिरिक्त|सावधानी))", text, re.IGNORECASE | re.DOTALL)
    if match and match.group(1).strip():
        sol = match.group(1).strip()
    else:
        sol = text.strip()
        
    sol = re.sub(r'[\n\*\-•]+', ' ', sol)
    sol = re.sub(r'\s+', ' ', sol).strip()
    
    sentences = re.split(r'(?<=[।\.!?:])\s+', sol)
    brief = " ".join(sentences[:3]).strip()
    
    if len(brief) > 250:
        brief = brief[:247] + "..."
    return brief

# ── Off-topic / greeting detector ─────────────────────────────────────────────
_GREETING_PATTERNS = {
    "hello", "hi", "hey", "helo", "hii", "hiii",
    "namaste", "namaskar", "नमस्ते", "नमस्कार",
    "good morning", "good evening", "good night", "good afternoon",
    "how are you", "aap kaise ho", "kya haal", "क्या हाल",
    "thanks", "thank you", "shukriya", "धन्यवाद", "शुक्रिया",
    "bye", "goodbye", "alvida", "अलविदा",
    "ok", "okay", "theek hai", "ठीक है", "accha", "अच्छा",
    "help", "madad", "मदद",
    "test", "testing",
}

_GREETING_RESPONSE = (
    "🌾 नमस्ते किसान भाई! 🙏\n\n"
    "मैं किसान AI सहायक हूँ। मैं केवल खेती और फसल से जुड़े सवालों का जवाब दे सकता हूँ।\n\n"
    "💡 आप मुझसे पूछ सकते हैं:\n"
    "• फसल में कीट या रोग की समस्या\n"
    "• खाद और उर्वरक की जानकारी\n"
    "• सिंचाई कब और कितनी करें\n"
    "• फसल उगाने का तरीका\n\n"
    "उदाहरण: \"आम में मिलीबग कीट लगे हैं\" या \"गेहूँ कैसे उगाएं\"\n\n"
    "📞 किसान हेल्पलाइन: 1800-180-1551 (निःशुल्क, सोम-शनि)\n"
    "🌐 आधिकारिक वेबसाइट: www.agricoop.nic.in"
)

# ── Privacy / restricted query detector ───────────────────────────────────────
_PRIVACY_PATTERNS = [
    # Who/what are you
    r"who (are|made|built|created|developed|trained|designed) you",
    r"what (are|is) you",
    r"aap kaun (ho|hain|hai)",
    r"tum kaun (ho|hain)",
    r"kisne banaya",
    r"kisne develop",
    r"kisne create",
    r"आप कौन", r"तुम कौन", r"किसने बनाया", r"किसने बनाई",
    r"किसने विकसित", r"किसने तैयार",
    # Which AI / model
    r"which (ai|model|llm|language model)",
    r"what (ai|model|llm)",
    r"konsa ai", r"kaunsa ai", r"konsa model", r"kaunsa model",
    r"कौन सा ai", r"कौन सा मॉडल", r"कौनसा ai",
    r"are you (chatgpt|gpt|gemini|claude|llama|ollama|openai|google|microsoft|meta)",
    r"(chatgpt|gpt-4|gemini|claude|llama|ollama) (hai|ho|hain|kya)",
    # Database / data questions
    r"(your|tera|aapka|tumhara) (database|data|dataset|training data)",
    r"(database|dataset|data) (kahan|kaha|se|mein|ka)",
    r"डेटाबेस (कहाँ|कहां|से|में|का|क्या)",
    r"डेटा (कहाँ|कहां|से|में|का|क्या)",
    r"(kitne|kitna|how many) (data|rows|records|entries|questions)",
    r"कितना डेटा", r"कितने रिकॉर्ड",
    # Creator / company
    r"(company|organization|organisation|team|developer|creator|owner)",
    r"(कंपनी|संस्था|टीम|डेवलपर|निर्माता|मालिक)",
    r"(built|made|created|developed) by",
    r"(द्वारा बनाया|द्वारा विकसित)",
    # Technology stack
    r"(technology|tech stack|framework|python|flask|ollama|langchain)",
    r"(टेक्नोलॉजी|फ्रेमवर्क|प्रोग्रामिंग)",
    r"how (do you|does this|are you) work",
    r"(kaise kaam|kaise work) (karta|karti|karte)",
    r"कैसे काम करते", r"कैसे काम करता",
    # Source code / API
    r"(source code|github|api key|api|endpoint)",
    r"(सोर्स कोड|एपीआई)",
    # Prompt / system
    r"(your prompt|system prompt|instructions|rules)",
    r"(प्रॉम्प्ट|सिस्टम प्रॉम्प्ट|निर्देश)",
]

_PRIVACY_RESPONSE = (
    "🔒 यह जानकारी साझा नहीं की जा सकती।\n\n"
    "मैं एक कृषि सहायक हूँ और केवल खेती से जुड़े सवालों का जवाब देने के लिए बना हूँ।\n\n"
    "💡 आप मुझसे पूछ सकते हैं:\n"
    "• फसल में कीट या रोग की समस्या\n"
    "• खाद और उर्वरक की जानकारी\n"
    "• सिंचाई कब और कितनी करें\n"
    "• फसल उगाने का तरीका\n\n"
    "उदाहरण: \"टमाटर में झुलसा रोग\" या \"धान में पानी कितना दें\"\n\n"
    "📞 किसान हेल्पलाइन: 1800-180-1551 (निःशुल्क, सोम-शनि)\n"
    "🌐 आधिकारिक वेबसाइट: www.agriwelfare.gov.in"
)

import random as _random

# ── Large pools for fallback response variety ──────────────────────────────────

_KARAN_POOL = [
    "मौसम परिवर्तन और अत्यधिक नमी से रोग फैलता है।",
    "मिट्टी में पोषक तत्वों की कमी से पौधे कमज़ोर होते हैं।",
    "अनुचित सिंचाई से जड़ों में सड़न होती है।",
    "कीटों का प्रकोप गर्म और आर्द्र मौसम में बढ़ता है।",
    "बीज उपचार न करने से रोग बीज से ही फैलता है।",
    "खेत में जल निकासी की कमी से फफूंद रोग होते हैं।",
    "अत्यधिक नाइट्रोजन उर्वरक से कीट आकर्षित होते हैं।",
    "फसल चक्र न अपनाने से मिट्टी में रोगाणु बढ़ते हैं।",
    "पुराने संक्रमित पौधों के अवशेष खेत में छोड़ने से रोग फैलता है।",
    "तापमान में अचानक बदलाव से पौधों की रोग प्रतिरोधक क्षमता घटती है।",
    "अत्यधिक वर्षा से मिट्टी में ऑक्सीजन की कमी होती है।",
    "सूखे की स्थिति में पौधे तनाव में आकर रोगग्रस्त होते हैं।",
    "संक्रमित औज़ारों के उपयोग से रोग एक पौधे से दूसरे में फैलता है।",
    "घनी बुवाई से हवा का संचार कम होता है और फफूंद बढ़ती है।",
    "मिट्टी का pH असंतुलित होने से पोषक तत्व अवशोषण बाधित होता है।",
    "जिंक, बोरोन या आयरन की कमी से पत्तियाँ पीली पड़ती हैं।",
    "कीट अंडे मिट्टी में सर्दियों में जीवित रहते हैं और वसंत में सक्रिय होते हैं।",
    "पड़ोसी खेत से हवा द्वारा रोगाणु आ सकते हैं।",
    "सिंचाई के पानी में रोगाणु होने से संक्रमण फैलता है।",
    "पौधों में पोटाश की कमी से रोग प्रतिरोधक क्षमता कम होती है।",
    "अत्यधिक कीटनाशक उपयोग से मित्र कीट नष्ट होते हैं।",
    "फास्फोरस की कमी से जड़ें कमज़ोर होती हैं।",
    "कैल्शियम की कमी से फल और पत्तियाँ सड़ने लगती हैं।",
    "मैग्नीशियम की कमी से पत्तियों में क्लोरोफिल कम होता है।",
    "अत्यधिक छाया में पौधे कमज़ोर होकर रोगग्रस्त होते हैं।",
    "बाढ़ के पानी से रोगाणु एक खेत से दूसरे खेत में पहुँचते हैं।",
    "कटाई के बाद खेत की सफाई न करने से अगली फसल प्रभावित होती है।",
    "असंतुलित खाद प्रबंधन से मिट्टी की उर्वरता घटती है।",
    "रोगग्रस्त पौध नर्सरी से लाने पर संक्रमण फैलता है।",
    "उच्च आर्द्रता में फफूंद के बीजाणु तेज़ी से फैलते हैं।",
    "कम तापमान में पौधों की वृद्धि रुक जाती है।",
    "अत्यधिक गर्मी में पत्तियाँ झुलस जाती हैं।",
    "मिट्टी में जैविक पदार्थ की कमी से उर्वरता घटती है।",
    "सल्फर की कमी से पौधों में प्रोटीन निर्माण बाधित होता है।",
    "मैंगनीज की कमी से पत्तियों पर धब्बे पड़ते हैं।",
    "कॉपर की कमी से पौधों में रोग प्रतिरोधक क्षमता कम होती है।",
    "मोलिब्डेनम की कमी से नाइट्रोजन स्थिरीकरण बाधित होता है।",
    "क्लोरीन की कमी से पत्तियाँ मुरझाती हैं।",
    "अत्यधिक लवणीय मिट्टी में पौधे पानी नहीं सोख पाते।",
    "भारी मिट्टी में जल भराव से जड़ें सड़ती हैं।",
    "हल्की बलुई मिट्टी में पोषक तत्व जल्दी बह जाते हैं।",
    "खरपतवार कीटों और रोगों के लिए आश्रय स्थल बनते हैं।",
    "एकल फसल प्रणाली से मिट्टी में विशेष रोगाणु बढ़ते हैं।",
    "रात के तापमान में गिरावट से ओस बनती है जो फफूंद को बढ़ावा देती है।",
    "तेज़ हवाओं से पत्तियाँ टूटती हैं और संक्रमण का रास्ता बनता है।",
    "ओलावृष्टि से पौधों पर घाव बनते हैं जहाँ रोगाणु प्रवेश करते हैं।",
    "पाले से पौधों की कोशिकाएँ नष्ट होती हैं।",
    "अम्लीय वर्षा से मिट्टी का pH बिगड़ता है।",
    "प्रदूषित सिंचाई जल से भारी धातुएँ मिट्टी में जमा होती हैं।",
    "कीटनाशकों के प्रति कीटों में प्रतिरोधक क्षमता विकसित हो जाती है।",
    "समय पर निराई न करने से खरपतवार पोषक तत्व छीन लेते हैं।",
    "अनुचित भंडारण से बीजों में फफूंद लग जाती है।",
    "पुराने बीजों की अंकुरण क्षमता कम होती है।",
    "बिना उपचारित बीज बोने से मिट्टीजनित रोग होते हैं।",
    "अत्यधिक यूरिया से मिट्टी अम्लीय होती है।",
    "DAP की अधिक मात्रा से जिंक की कमी हो सकती है।",
    "पोटाश की कमी से फल छोटे और कम मीठे होते हैं।",
    "बोरोन की कमी से फूल और फल गिरते हैं।",
    "आयरन की कमी से नई पत्तियाँ पीली पड़ती हैं।",
    "नेमाटोड (सूत्रकृमि) जड़ों को नुकसान पहुँचाते हैं।",
    "दीमक जड़ों और तने को खोखला कर देती है।",
    "चूहे फसल को जड़ से काट देते हैं।",
    "पक्षी फल और बीज नष्ट करते हैं।",
    "माहू (एफिड) पत्तियों का रस चूसकर वायरस फैलाते हैं।",
    "सफेद मक्खी वायरल रोगों की वाहक होती है।",
    "थ्रिप्स फूलों और कोमल पत्तियों को नुकसान पहुँचाते हैं।",
    "माइट्स पत्तियों की निचली सतह से रस चूसते हैं।",
    "मिलीबग पौधों के रस को चूसकर उन्हें कमज़ोर करते हैं।",
    "स्केल कीट तने और पत्तियों पर चिपककर नुकसान करते हैं।",
    "तना बेधक तने के अंदर घुसकर पौधे को नष्ट करता है।",
    "फल बेधक फल के अंदर घुसकर उसे सड़ा देता है।",
    "पत्ती लपेटक पत्तियों को लपेटकर अंदर से खाता है।",
    "कटुआ इल्ली रात में पौधों को जड़ से काटती है।",
    "सैनिक कीट झुंड में आकर पूरी फसल चट कर जाते हैं।",
    "ग्रासहॉपर (टिड्डी) पत्तियाँ और तने खा जाते हैं।",
    "जैसिड पत्तियों का रस चूसकर उन्हें पीला करते हैं।",
    "भूरा फुदका धान की फसल को जड़ से नष्ट करता है।",
    "गंधी बग धान के दानों का रस चूसता है।",
    "हिस्पा धान की पत्तियों को खुरचकर खाता है।",
    "ब्लास्ट रोग ठंडे और आर्द्र मौसम में तेज़ी से फैलता है।",
    "झुलसा रोग बारिश के बाद अचानक फैलता है।",
    "उकठा रोग मिट्टीजनित फफूंद से होता है।",
    "मोज़ेक वायरस कीटों द्वारा फैलता है।",
    "पाउडरी मिल्ड्यू शुष्क मौसम में अधिक होती है।",
    "डाउनी मिल्ड्यू आर्द्र मौसम में फैलती है।",
    "रतुआ रोग हवा द्वारा फैलता है।",
    "टिक्का रोग मूंगफली में पत्ती धब्बे बनाता है।",
    "बकानी रोग धान में बीज से फैलता है।",
    "खैरा रोग जिंक की कमी से होता है।",
    "कैंकर रोग बैक्टीरिया से होता है।",
    "गलन रोग अत्यधिक नमी में होता है।",
    "सड़न रोग भंडारण में अनुचित नमी से होता है।",
    "विल्ट रोग मिट्टी में फ्यूज़ेरियम फफूंद से होता है।",
    "लीफ कर्ल वायरस सफेद मक्खी द्वारा फैलता है।",
    "बंट रोग गेहूँ में बीज उपचार न करने से होता है।",
    "स्मट रोग बीज और मिट्टी दोनों से फैलता है।",
    "एन्थ्रेक्नोज़ रोग आर्द्र मौसम में फलों पर होता है।",
    "बोट्राइटिस ब्लाइट ठंडे और नम मौसम में होता है।",
    "फाइटोफ्थोरा रोग जल भराव वाली मिट्टी में होता है।",
    "राइज़ोक्टोनिया रोग मिट्टी में लंबे समय तक जीवित रहता है।",
]

_SAVDHAN_POOL = [
    "रसायन उपयोग करते समय दस्ताने और मास्क पहनें।",
    "छिड़काव सुबह या शाम को करें, तेज़ धूप में नहीं।",
    "बच्चों और पालतू जानवरों को छिड़काव क्षेत्र से दूर रखें।",
    "दवाई की निर्धारित मात्रा से अधिक उपयोग न करें।",
    "कीटनाशक को खाद्य पदार्थों के पास न रखें।",
    "खाली कीटनाशक डिब्बों को जलाएँ या गहरे गड्ढे में दबाएँ।",
    "छिड़काव के बाद हाथ-मुँह साबुन से अच्छी तरह धोएँ।",
    "कीटनाशक को बच्चों की पहुँच से दूर बंद जगह रखें।",
    "एक ही कीटनाशक बार-बार उपयोग न करें, बदलते रहें।",
    "फसल कटाई से कम से कम 15 दिन पहले छिड़काव बंद करें।",
    "तेज़ हवा में छिड़काव न करें।",
    "बारिश से पहले छिड़काव न करें।",
    "दो अलग-अलग कीटनाशकों को बिना जानकारी के न मिलाएँ।",
    "कीटनाशक को सीधे जल स्रोतों के पास न फेंकें।",
    "छिड़काव के दौरान धूम्रपान न करें।",
    "कीटनाशक को मुँह से न चखें।",
    "पुराने और एक्सपायर्ड कीटनाशक का उपयोग न करें।",
    "कीटनाशक खरीदते समय लाइसेंसी दुकान से ही खरीदें।",
    "नकली कीटनाशक से बचें, पैकेट पर लाइसेंस नंबर जाँचें।",
    "जैविक खेती में रासायनिक कीटनाशक का उपयोग न करें।",
    "मधुमक्खियों के सक्रिय समय में कीटनाशक न छिड़कें।",
    "फूल आने के समय कीटनाशक छिड़काव से बचें।",
    "सुरक्षात्मक चश्मा पहनकर छिड़काव करें।",
    "छिड़काव के बाद कपड़े अलग से धोएँ।",
    "कीटनाशक के संपर्क में आने पर तुरंत डॉक्टर से मिलें।",
    "खेत में काम करने वाले मज़दूरों को सुरक्षा उपकरण दें।",
    "गर्भवती महिलाएँ कीटनाशक छिड़काव से दूर रहें।",
    "कीटनाशक को मूल पैकेट में ही रखें, दूसरे बर्तन में नहीं।",
    "छिड़काव यंत्र को उपयोग के बाद अच्छी तरह साफ करें।",
    "लीक हो रहे स्प्रेयर का उपयोग न करें।",
    "मिट्टी परीक्षण के बिना अंधाधुंध खाद न डालें।",
    "यूरिया को बारिश से पहले न डालें, बह जाएगा।",
    "DAP को बुवाई के समय मिट्टी में मिलाएँ।",
    "पोटाश को सिंचाई के साथ दें।",
    "खाद की अधिक मात्रा से मिट्टी खराब होती है।",
    "जैविक खाद को कच्चा न डालें, पका हुआ उपयोग करें।",
    "नीम खली को मिट्टी में अच्छी तरह मिलाएँ।",
    "हरी खाद को मिट्टी में पलटने के बाद 15 दिन प्रतीक्षा करें।",
    "वर्मीकम्पोस्ट को सीधे धूप में न रखें।",
    "बायोफर्टिलाइज़र को रासायनिक खाद के साथ न मिलाएँ।",
    "सिंचाई का पानी साफ और प्रदूषण मुक्त होना चाहिए।",
    "ड्रिप सिंचाई में फिल्टर नियमित साफ करें।",
    "स्प्रिंकलर से सिंचाई रात में न करें।",
    "जल भराव से बचने के लिए खेत में नाली बनाएँ।",
    "अत्यधिक सिंचाई से जड़ें सड़ती हैं।",
    "कम सिंचाई से पौधे सूखते हैं।",
    "बीज उपचार के बाद बीज को छाया में सुखाएँ।",
    "उपचारित बीज को हाथ से न छुएँ।",
    "बीज को नम जगह पर न रखें।",
    "प्रमाणित बीज ही खरीदें।",
    "खेत में काम के बाद हाथ धोकर ही खाना खाएँ।",
    "कीटनाशक के नज़दीक खाना न खाएँ।",
    "खेत में काम करते समय पानी की बोतल साथ रखें।",
    "गर्मी में दोपहर 12 से 3 बजे खेत में काम न करें।",
    "खेत में काम करते समय टोपी पहनें।",
    "साँप और बिच्छू से सावधान रहें।",
    "खेत में काम करते समय जूते पहनें।",
    "कटाई के औज़ार तेज़ और साफ रखें।",
    "कटाई के बाद फसल को नम जगह पर न रखें।",
    "भंडारण से पहले अनाज को अच्छी तरह सुखाएँ।",
    "भंडारण में नमी 12% से कम रखें।",
    "गोदाम में चूहों से बचाव करें।",
    "भंडारण में कीटनाशक धुआँ (फ्यूमिगेशन) करें।",
    "पुराने अनाज को नए के साथ न मिलाएँ।",
    "बाज़ार में फसल बेचने से पहले नमी जाँचें।",
    "फसल को बारिश में भीगने से बचाएँ।",
    "ट्रैक्टर चलाते समय सीट बेल्ट लगाएँ।",
    "कृषि यंत्रों की नियमित सर्विसिंग करें।",
    "बिजली के तारों के पास काम करते समय सावधान रहें।",
    "सिंचाई पंप को बारिश में बंद रखें।",
    "खेत में आग न जलाएँ।",
    "पराली जलाने से मिट्टी के जीवाणु नष्ट होते हैं।",
    "खेत की मेड़ पर पेड़ लगाएँ।",
    "मिट्टी कटाव रोकने के लिए खेत में बाँध बनाएँ।",
    "रासायनिक खाद को बीज के सीधे संपर्क में न रखें।",
    "खाद को मिट्टी में अच्छी तरह मिलाएँ।",
    "फसल बीमा ज़रूर करवाएँ।",
    "कृषि विभाग की सलाह के बिना नई दवाई न आज़माएँ।",
    "किसान कॉल सेंटर 1800-180-1551 पर निःशुल्क सलाह लें।",
    "नज़दीकी KVK से मिट्टी परीक्षण करवाएँ।",
    "प्रशिक्षण शिविरों में भाग लें।",
    "अन्य किसानों के अनुभव से सीखें।",
    "मौसम पूर्वानुमान देखकर खेती की योजना बनाएँ।",
    "फसल कटाई के बाद खेत की गहरी जुताई करें।",
    "गर्मी में खेत को खाली छोड़ें ताकि कीट नष्ट हों।",
    "फेरोमोन ट्रैप लगाकर कीटों की निगरानी करें।",
    "पीले चिपचिपे ट्रैप से सफेद मक्खी पकड़ें।",
    "प्रकाश प्रपंच से रात के कीट पकड़ें।",
    "जैविक कीटनाशक जैसे नीम तेल का उपयोग करें।",
    "ट्राइकोडर्मा से बीज उपचार करें।",
    "स्यूडोमोनास से मिट्टी उपचार करें।",
    "बायोपेस्टिसाइड का उपयोग बढ़ाएँ।",
    "एकीकृत कीट प्रबंधन (IPM) अपनाएँ।",
    "रोगरोधी किस्मों का चयन करें।",
    "उचित दूरी पर बुवाई करें।",
    "समय पर बुवाई करें।",
    "उचित गहराई पर बीज बोएँ।",
    "खेत की तैयारी में गहरी जुताई करें।",
    "फसल की नियमित निगरानी करें और समस्या जल्दी पहचानें।",
    "कृषि विशेषज्ञ की सलाह के बिना नई दवाई न आज़माएँ।",
]

def _pick_random(pool: list, n: int) -> list:
    """Pick n unique random items from pool, shuffled differently each call."""
    return _random.sample(pool, min(n, len(pool)))

def _pick_relevant(pool: list, query_text: str, n: int = 3) -> list:
    """Pick n most relevant items from pool using TF-IDF cosine similarity."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity as _cos_sim
    import numpy as np

    corpus = [query_text] + pool
    try:
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1)
        mat = vec.fit_transform(corpus)
        sims = _cos_sim(mat[0:1], mat[1:])[0]
        top_idx = np.argsort(sims)[::-1][:n]
        return [pool[i] for i in top_idx]
    except Exception:
        return _random.sample(pool, min(n, len(pool)))

import re as _re2
_PRIVACY_COMPILED = [_re2.compile(p, _re2.IGNORECASE) for p in _PRIVACY_PATTERNS]

def is_privacy_query(query: str) -> bool:
    """Returns True if query asks about system internals, creator, AI model, database."""
    q = query.strip()
    return any(p.search(q) for p in _PRIVACY_COMPILED)

def is_off_topic(query: str) -> bool:
    """Returns True if query is a greeting or clearly non-farming."""
    q = query.strip().lower()
    # Exact match
    if q in _GREETING_PATTERNS:
        return True
    # Very short with no farming keywords
    farming_hints = ["फसल", "कीट", "रोग", "खाद", "पानी", "बीज", "पेड़", "पौधा",
                     "crop", "pest", "disease", "fertilizer", "water", "seed",
                     "ugana", "kheti", "fasal", "keede", "rog", "khad"]
    if len(q.split()) <= 2 and not any(h in q for h in farming_hints):
        return True
    return False

# ── Hindi digit → English digit conversion ────────────────────────────────────
_HINDI_DIGITS = str.maketrans('०१२३४५६७८९', '0123456789')

def hindi_to_english_numbers(text: str) -> str:
    """Replace Hindi/Devanagari digits with English digits."""
    return text.translate(_HINDI_DIGITS)


# ── Dataset solution text cleaner ─────────────────────────────────────────────
import re as _re

_ABBREV_MAP = [
    (_re.compile(r'कि0ग्रा0'),   'किलोग्राम'),
    (_re.compile(r'कि\.ग्रा\.'), 'किलोग्राम'),
    (_re.compile(r'ग्रा0'),      'ग्राम'),
    (_re.compile(r'मि0ली0'),     'मिली'),
    (_re.compile(r'मि\.ली\.'),   'मिली'),
    (_re.compile(r'ली0'),        'लीटर'),
    (_re.compile(r'मि0'),        'मिली'),
    (_re.compile(r'प्रति0'),     'प्रति'),
    (_re.compile(r'हेक्ट0'),     'हेक्टेयर'),
    (_re.compile(r'हे0'),        'हेक्टेयर'),
    (_re.compile(r'डब्लू0पी0'),  'WP'),
    (_re.compile(r'ई0सी0'),      'EC'),
    (_re.compile(r'एस0सी0'),     'SC'),
    (_re.compile(r'एस0एल0'),     'SL'),
    (_re.compile(r'डब्लू0जी0'),  'WG'),
    (_re.compile(r'मि0ली0'),     'मिली'),
    (_re.compile(r'\s{2,}'),     ' '),       # collapse multiple spaces
]

def clean_solution_text(text: str) -> str:
    """Fix garbled dataset abbreviations before sending to LLM or displaying."""
    for pattern, replacement in _ABBREV_MAP:
        text = pattern.sub(replacement, text)
    return text.strip()


# ─────────────────────────────────────────────────────────────────────────────
# PROMPT BUILDERS
# ─────────────────────────────────────────────────────────────────────────────

def _context_block(context_rows: list) -> str:
    """Format dataset rows into clean Hindi context. Never leaks template markers."""
    if not context_rows:
        return ""
    lines = []
    for i, row in enumerate(context_rows, 1):
        sim  = row.get("similarity", 0)
        lines.append(
            f"संदर्भ {i}\n"
            f"समस्या: {clean_solution_text(row['problem'])}\n"
            f"समाधान: {clean_solution_text(row['solution'])}\n"
            f"फसल: {row['crop']}"
        )
    return "\n\n".join(lines)


def build_prompt_with_context(query: str, crop: str | None, intent: str, context_rows: list) -> str:
    """Used when dataset has relevant matches (similarity threshold cleared)."""
    crop_text   = crop or "आपकी फसल"
    intent_text = INTENT_LABELS.get(intent, intent)
    ctx         = _context_block(context_rows)

    if intent in ("cultivation", "general"):
        return build_prompt_cultivation(query, crop_text, ctx)

    return f"""You are an expert Krishi Sahayak helping Indian farmers. Answer ONLY in simple Hindi.

किसान की समस्या: {query}
फसल: {crop_text}
समस्या का प्रकार: {intent_text}

विशेषज्ञ डेटाबेस से संदर्भ:
{ctx}

STRICT RULES:
- Answer ONLY in Hindi using Devanagari script. Do NOT write any word in Roman/English letters except chemical/product names and numbers.
- Use the database context as primary source. Enhance it with your knowledge.
- Do NOT start with "धन्यवाद", "नमस्ते", "मैं यहाँ हूँ" or any greeting.
- Do NOT mention "dataset", "database", "AI", "system", "संदर्भ", "जानकारी के अनुसार"
- Do NOT echo back the user's question or include placeholder text
- ALWAYS write numbers and quantities using English digits (0-9), e.g. 5 ml, 50 gram, 15 liter — NEVER use Hindi/Devanagari digits like ५, ५०, १५
- Start DIRECTLY with: 🌱 समस्या:

OUTPUT FORMAT (follow exactly):
🌱 समस्या:
[2-3 lines explaining the crop problem clearly]

🔍 कारण:
• [कारण 1]
• [कारण 2]
• [कारण 3 यदि प्रासंगिक हो]

🛠 समाधान:
[Numbered steps. Include chemical names + dosage from context exactly. Add timing.]

⚠ सावधानियां:
• [सावधानी 1]
• [सावधानी 2]
• [सावधानी 3]

💡 अतिरिक्त सलाह:
[Provide 1-2 lines of HIGHLY RELEVANT and SHORT extra advice related ONLY to the specific query]

🌱 समस्या: से शुरू करें:"""


def build_prompt_llm_only(query: str, crop: str | None, intent: str) -> str:
    """Used when no dataset row clears threshold — pure expert LLM knowledge."""
    crop_text   = crop or "आपकी फसल"
    intent_text = INTENT_LABELS.get(intent, intent)

    if intent in ("cultivation", "general"):
        return build_prompt_cultivation(query, crop_text, "")

    return f"""You are an expert Krishi Sahayak (agricultural officer) with 20+ years experience helping Indian farmers.
Answer ONLY in simple Hindi (Devanagari script).

किसान की समस्या: {query}
फसल: {crop_text}
विषय: {intent_text}

STRICT RULES:
- Answer ONLY in Hindi using Devanagari script. Do NOT write any word in Roman/English letters except chemical/product names and numbers.
- Do NOT start with "धन्यवाद", "नमस्ते", or any greeting
- Do NOT echo the question back
- Do NOT use placeholder text like [विशेषज्ञ ज्ञान 1] or similar
- Be specific: include real chemical names, dosages, timings for Indian farmers
- ALWAYS write numbers and quantities using English digits (0-9), e.g. 5 ml, 50 gram, 15 liter — NEVER use Hindi/Devanagari digits like ५, ५०, १५
- Start DIRECTLY with: 🌱 समस्या:

OUTPUT FORMAT (follow exactly):
🌱 समस्या:
[2-3 lines explaining the crop problem]

🔍 कारण:
• [कारण 1]
• [कारण 2]
• [कारण 3 यदि प्रासंगिक हो]

🛠 समाधान:
[Numbered practical steps with dosage and timing]

⚠ सावधानियां:
• [सावधानी 1]
• [सावधानी 2]
• [सावधानी 3]

💡 अतिरिक्त सलाह:
[2-3 lines of expert practical insight]

🌱 समस्या: से शुरू करें:"""


def build_prompt_cultivation(query: str, crop_text: str, ctx: str) -> str:
    """Used for general queries and specific cultivation questions (like pruning, sowing)."""
    from datetime import date
    today = date.today()
    month_names = {
        1:"जनवरी", 2:"फरवरी", 3:"मार्च", 4:"अप्रैल",
        5:"मई", 6:"जून", 7:"जुलाई", 8:"अगस्त",
        9:"सितंबर", 10:"अक्टूबर", 11:"नवंबर", 12:"दिसंबर"
    }
    current_month = month_names[today.month]

    return f"""You are an expert Krishi Sahayak helping Indian farmers.
Answer ONLY in Hindi using Devanagari script.

वर्तमान महीना: {current_month}
किसान का सवाल: {query}
फसल: {crop_text}

विशेषज्ञ डेटाबेस से संदर्भ:
{ctx}

STRICT RULES:
1. Answer EXACTLY what the user is asking. Do not provide unnecessary or unrequested information.
2. If the user asks a SPECIFIC question (e.g., "when to prune", "how much water", "when to sow"), provide a SHORT, direct answer using the provided context if available. Do NOT output a full cultivation guide.
3. ONLY IF the user explicitly asks for complete cultivation details (e.g., "खेती कैसे करें?", "पूरी जानकारी दें"), provide a complete step-by-step guide.
4. Keep the answer concise to reduce reading time.
5. ALWAYS write numbers using English digits (0-9), e.g. 5 kg, 100 liter.
6. Start DIRECTLY with: 🌾 जानकारी:

OUTPUT FORMAT:
Provide the answer in a clear, easy-to-read format. Use these headings:

🌾 जानकारी:
[Direct 2-3 line answer to the specific question asked]

🛠 मुख्य बिंदु:
• [Practical point 1]
• [Practical point 2]
• [Practical point 3 if needed]

💡 अतिरिक्त सलाह:
[1-2 lines of HIGHLY RELEVANT extra advice ONLY about the specific topic asked, related to {current_month}]

🌾 जानकारी: से शुरू करें:"""


# ─────────────────────────────────────────────────────────────────────────────
# OLLAMA STREAMING CALLER
# ─────────────────────────────────────────────────────────────────────────────

def stream_ollama(prompt: str):
    """
    Generator: yields text chunks as they arrive from Ollama streaming API.
    Yields None on error/timeout.
    """
    payload = {
        "model":  MODEL,
        "prompt": prompt,
        "stream": True,
        "options": {
            "temperature":    0.20,
            "top_p":          0.90,
            "num_predict":    1200,
            "repeat_penalty": 1.1,
        },
    }
    try:
        with requests.post(OLLAMA_URL, json=payload, stream=True, timeout=180) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if line:
                    try:
                        chunk = json.loads(line)
                        token = chunk.get("response", "")
                        if token:
                            yield token
                        if chunk.get("done"):
                            break
                    except json.JSONDecodeError:
                        continue
    except Exception as e:
        print(f"❌ Ollama streaming error: {e}", flush=True)
        yield None  # sentinel: caller knows to use fallback


def query_ollama_full(prompt: str) -> str | None:
    """Non-streaming call — used only in fallback path."""
    payload = {
        "model":  MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature":    0.20,
            "top_p":          0.90,
            "num_predict":    1200,
            "repeat_penalty": 1.1,
        },
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=180)
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# DATASET FALLBACK (when Ollama offline)
# ─────────────────────────────────────────────────────────────────────────────

def fallback_response(query: str, crop: str | None, intent: str, rows: list) -> str:
    crop_text   = crop or "आपकी फसल"
    intent_text = INTENT_LABELS.get(intent, "सामान्य")

    if not rows:
        return (
            f"🌱 समस्या:\n"
            f"{crop_text} में {intent_text} की समस्या है।\n\n"
            f"🔍 कारण:\n"
            f"• इस विषय पर विस्तृत जानकारी डेटाबेस में अभी उपलब्ध नहीं है।\n\n"
            f"🛠 समाधान:\n"
            f"• अपने नज़दीकी कृषि विज्ञान केंद्र (KVK) से संपर्क करें।\n"
            f"• Kisan Call Center: 1800-180-1551 (निःशुल्क)\n\n"
            f"⚠ सावधानियां:\n"
            f"• बिना जानकारी के कोई रसायन उपयोग न करें।\n\n"
            f"💡 अतिरिक्त सलाह:\n"
            f"• जिले के कृषि अधिकारी से मिट्टी परीक्षण करवाएं।"
        )

    best  = rows[0]

    # Build context text for relevance matching
    context_text = f"{query} {best['problem']} {best['solution']}"

    # Pick 3 most relevant कारण and सावधानियां via cosine similarity
    karan   = _pick_relevant(_KARAN_POOL,   context_text, 3)
    savdhan = _pick_relevant(_SAVDHAN_POOL, context_text, 3)

    karan_text   = "\n".join(f"• {k}" for k in karan)
    savdhan_text = "\n".join(f"• {s}" for s in savdhan)

    return (
        f"🌱 समस्या:\n"
        f"{crop_text} में {intent_text} की समस्या पाई गई है।\n"
        f"{clean_solution_text(best['problem'])}\n\n"
        f"🔍 कारण:\n"
        f"{karan_text}\n\n"
        f"🛠 समाधान:\n"
        f"{clean_solution_text(best['solution'])}\n\n"
        f"⚠ सावधानियां:\n"
        f"{savdhan_text}\n\n"
        f"💡 अतिरिक्त सलाह:\n"
        f"• नियमित फसल निरीक्षण करें।\n"
        f"• Kisan Call Center 1800-180-1551 पर निःशुल्क सलाह लें।"
    )


# ─────────────────────────────────────────────────────────────────────────────
# SUGGESTION GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

def generate_suggestions(crop: str | None, intent: str, rows: list) -> list:
    suggestions: list = []
    if crop:
        intent_map = {
            "pest":       f"{crop} में रोग की समस्या",
            "disease":    f"{crop} में खाद की कमी",
            "fertilizer": f"{crop} में पानी कब दें",
            "irrigation": f"{crop} में कीट नियंत्रण",
            "growth":     f"{crop} की उपज कैसे बढ़ाएं",
        }
        for k, v in intent_map.items():
            if k != intent and len(suggestions) < 2:
                suggestions.append(v)

    general = {
        "pest":        ["जैविक कीटनाशक के उपाय", "कीट प्रबंधन के घरेलू तरीके"],
        "disease":     ["फसल रोग की पहचान कैसे करें", "जैविक रोग नियंत्रण"],
        "fertilizer":  ["मिट्टी परीक्षण कैसे करें", "जैविक खाद बनाने की विधि"],
        "irrigation":  ["ड्रिप सिंचाई के फायदे", "पानी बचाने के तरीके"],
        "growth":      ["फसल की देखभाल के टिप्स", "उपज बढ़ाने के आसान उपाय"],
        "cultivation": ["बीज उपचार कैसे करें", "मिट्टी परीक्षण कैसे करें"],
        "general":     ["मिट्टी की जांच कैसे करें", "जैविक खेती के फायदे"],
    }
    for s in general.get(intent, []):
        if len(suggestions) < 3:
            suggestions.append(s)

    fallbacks = ["मिट्टी की जांच कैसे करें", "जैविक खेती के फायदे", "फसल चक्र क्या है"]
    for f in fallbacks:
        if len(suggestions) < 3:
            suggestions.append(f)
    return suggestions[:4]


# ─────────────────────────────────────────────────────────────────────────────
# STREAMING PIPELINE — yields Server-Sent Event strings
# ─────────────────────────────────────────────────────────────────────────────

def stream_answer(user_query: str):
    """
    Generator that yields SSE-formatted strings:
      data: {"type":"meta", ...}        — once, first
      data: {"type":"token","text":"…"} — many
      data: {"type":"done", ...}        — once, last
      data: {"type":"error","text":"…"} — on failure
    """
    ds = get_dataset()

    # ── Privacy guard — block system/internal questions ───────────────────
    if is_privacy_query(user_query):
        print(f"🔒 Privacy query blocked: '{user_query}'", flush=True)
        meta = {
            "type": "meta", "crop": None, "intent": "general",
            "source": "system", "rows_found": 0, "top_similarity": 0,
            "suggestions": ["आम में मिलीबग कीट", "गेहूँ कैसे उगाएं", "टमाटर में रोग", "धान में पानी कितना दें"],
        }
        yield f"data: {json.dumps(meta, ensure_ascii=False)}\n\n"
        yield f"data: {json.dumps({'type': 'token', 'text': _PRIVACY_RESPONSE}, ensure_ascii=False)}\n\n"
        yield f"data: {json.dumps({'type': 'done', 'source': 'system'}, ensure_ascii=False)}\n\n"
        return

    # ── Off-topic / greeting guard ─────────────────────────────────────────
    if is_off_topic(user_query):
        print(f"👋 Off-topic query detected: '{user_query}' → instant reply", flush=True)
        meta = {
            "type": "meta", "crop": None, "intent": "general",
            "source": "system", "rows_found": 0, "top_similarity": 0,
            "suggestions": ["आम में मिलीबग कीट", "गेहूँ कैसे उगाएं", "टमाटर में रोग", "धान में पानी कितना दें"],
        }
        yield f"data: {json.dumps(meta, ensure_ascii=False)}\n\n"
        yield f"data: {json.dumps({'type': 'token', 'text': _GREETING_RESPONSE}, ensure_ascii=False)}\n\n"
        yield f"data: {json.dumps({'type': 'done', 'source': 'system'}, ensure_ascii=False)}\n\n"
        return

    crop_hindi, crop_alias = ds.detect_crop(user_query)
    intent = ds.detect_intent(user_query)
    rows   = ds.retrieve(user_query, crop_hindi, intent, top_k=5)

    # ── Debug log ──────────────────────────────────────────────────────────
    print(f"\n{'─'*60}", flush=True)
    print(f"❓ Query    : {user_query}", flush=True)
    print(f"🌱 Crop     : {crop_hindi or 'N/A'} (alias: {crop_alias or 'N/A'})", flush=True)
    print(f"🔍 Intent   : {intent}", flush=True)
    print(f"📊 Matches  : {len(rows)} rows (threshold-filtered)", flush=True)

    top_sim   = rows[0]["similarity"] if rows else 0.0
    top_score = rows[0]["score"]      if rows else 0.0

    # Tier decision uses boosted score (includes crop+intent bonus)
    if top_score >= 0.85:
        tier = "🟢 DATASET-DIRECT"
        tier_desc = f"Exact match (score={top_score:.3f}) — bypassing LLM for INSTANT response"
        source_tier = "dataset"
    elif top_score >= 0.50:
        tier = "🟡 LLM + CONTEXT"
        tier_desc = f"Good match (score={top_score:.3f}) — LLM enriched with dataset context"
        source_tier = "llm+dataset"
    else:
        tier = "🔴 LLM-ONLY"
        tier_desc = f"No/weak match (score={top_score:.3f}) — pure LLM knowledge"
        source_tier = "llm"

    print(f"🎯 Source   : {tier}", flush=True)
    print(f"   {tier_desc}", flush=True)
    print(f"   top_sim={top_sim:.3f}  top_score={top_score:.3f}", flush=True)
    for i, r in enumerate(rows, 1):
        print(f"  [{i}] sim={r['similarity']:.3f} score={r['score']:.3f} | {r['problem'][:60]}", flush=True)
    print(f"{'─'*60}", flush=True)

    # ── Send metadata to frontend ──────────────────────────────────────────
    meta = {
        "type":           "meta",
        "crop":           crop_hindi,
        "intent":         INTENT_LABELS.get(intent, intent),
        "source":         source_tier,
        "rows_found":     len(rows),
        "top_similarity": top_sim,
        "suggestions":    generate_suggestions(crop_hindi, intent, rows),
    }
    yield f"data: {json.dumps(meta, ensure_ascii=False)}\n\n"

    # ── TIER 1: DATASET-DIRECT — skip LLM entirely ────────────────────────
    if source_tier == "dataset":
        print(f"⚡ DATASET-DIRECT: skipping LLM", flush=True)
        fb = fallback_response(user_query, crop_hindi, intent, rows)
        fb = hindi_to_english_numbers(fb)
        yield f"data: {json.dumps({'type': 'token', 'text': fb}, ensure_ascii=False)}\n\n"
        yield f"data: {json.dumps({'type': 'done', 'source': 'dataset'}, ensure_ascii=False)}\n\n"
        return

    # ── TIER 2 & 3: LLM (with or without context) ─────────────────────────
    if source_tier == "llm+dataset":
        prompt = build_prompt_with_context(user_query, crop_hindi, intent, rows)
    else:
        prompt = build_prompt_llm_only(user_query, crop_hindi, intent)

    # ── Stream tokens ──────────────────────────────────────────────────────
    full_text = []
    error_hit = False

    for token in stream_ollama(prompt):
        if token is None:
            error_hit = True
            break
        token = hindi_to_english_numbers(token)
        full_text.append(token)
        yield f"data: {json.dumps({'type': 'token', 'text': token}, ensure_ascii=False)}\n\n"

    # ── Fallback if Ollama failed ──────────────────────────────────────────
    if error_hit or not full_text:
        fb = fallback_response(user_query, crop_hindi, intent, rows)
        yield f"data: {json.dumps({'type': 'token', 'text': fb}, ensure_ascii=False)}\n\n"
        yield f"data: {json.dumps({'type': 'done', 'source': 'dataset'}, ensure_ascii=False)}\n\n"
        return

    generated_text = "".join(full_text)

    # Save to Excel if this was a purely LLM generated response
    if source_tier == "llm":
        brief_sol = extract_brief_solution(generated_text)
        async_save_problem(query=user_query, brief_solution=brief_sol, category=INTENT_LABELS.get(intent, intent))

    yield f"data: {json.dumps({'type': 'done', 'source': 'llm'}, ensure_ascii=False)}\n\n"


# ─────────────────────────────────────────────────────────────────────────────
# NON-STREAMING PIPELINE (kept for /chat compatibility)
# ─────────────────────────────────────────────────────────────────────────────

def get_answer(user_query: str) -> dict:
    ds = get_dataset()
    if is_privacy_query(user_query):
        return {
            "response": _PRIVACY_RESPONSE, "crop": None,
            "intent": "general", "source": "system",
            "rows_found": 0, "suggestions": ["आम में मिलीबग कीट", "गेहूँ कैसे उगाएं"],
            "top_similarity": 0,
        }
    if is_off_topic(user_query):
        return {
            "response": _GREETING_RESPONSE, "crop": None,
            "intent": "general", "source": "system",
            "rows_found": 0, "suggestions": ["आम में मिलीबग कीट", "गेहूँ कैसे उगाएं"],
            "top_similarity": 0,
        }
    crop_hindi, crop_alias = ds.detect_crop(user_query)
    intent = ds.detect_intent(user_query)
    rows   = ds.retrieve(user_query, crop_hindi, intent, top_k=5)

    print(f"\n{'─'*60}", flush=True)
    print(f"❓ Query    : {user_query}", flush=True)
    print(f"🌱 Crop     : {crop_hindi or 'N/A'}", flush=True)
    print(f"🔍 Intent   : {intent}", flush=True)
    print(f"📊 Matches  : {len(rows)} rows", flush=True)
    
    top_sim   = rows[0]["similarity"] if rows else 0.0
    top_score = rows[0]["score"]      if rows else 0.0

    if top_score >= 0.85:
        tier = "🟢 DATASET-DIRECT"
        tier_desc = f"Exact match (score={top_score:.3f}) — bypassing LLM for INSTANT response"
        source_tier = "dataset"
    elif top_score >= 0.50:
        tier = "🟡 LLM + CONTEXT"
        tier_desc = f"Good match (score={top_score:.3f})"
        source_tier = "llm+dataset"
    else:
        tier = "🔴 LLM-ONLY"
        tier_desc = f"No/weak match (score={top_score:.3f})"
        source_tier = "llm"

    print(f"🎯 Source   : {tier}", flush=True)
    print(f"   {tier_desc}", flush=True)
    print(f"   top_sim={top_sim:.3f}  top_score={top_score:.3f}", flush=True)
    for i, r in enumerate(rows, 1):
        print(f"  [{i}] sim={r['similarity']:.3f} score={r['score']:.3f} | {r['problem'][:60]}", flush=True)
    print(f"{'─'*60}", flush=True)

    if source_tier == "dataset":
        llm_response = fallback_response(user_query, crop_hindi, intent, rows)
        llm_response = hindi_to_english_numbers(llm_response)
        source = "dataset"
    else:
        if source_tier == "llm+dataset":
            prompt = build_prompt_with_context(user_query, crop_hindi, intent, rows)
        else:
            prompt = build_prompt_llm_only(user_query, crop_hindi, intent)
        llm_response = query_ollama_full(prompt)
        if not llm_response:
            llm_response = fallback_response(user_query, crop_hindi, intent, rows)
            source = "dataset"
        else:
            llm_response = hindi_to_english_numbers(llm_response)
            source = source_tier
            
        # Log unanswered problem if we used LLM
        if source_tier == "llm" and source == "llm":
            brief_sol = extract_brief_solution(llm_response)
            async_save_problem(query=user_query, brief_solution=brief_sol, category=INTENT_LABELS.get(intent, intent))

    return {
        "response":       llm_response,
        "crop":           crop_hindi,
        "intent":         INTENT_LABELS.get(intent, intent),
        "source":         source,
        "rows_found":     len(rows),
        "suggestions":    generate_suggestions(crop_hindi, intent, rows),
        "top_similarity": top_sim,
    }
