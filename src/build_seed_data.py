# -*- coding: utf-8 -*-
"""
build_seed_data.py

Generates the two core data files for this project:

    data/raw/questions.csv
    data/knowledge_base/knowledge_base.csv

Design rationale
-----------------
Rather than using unrelated question sets per language, this script defines
15 underlying facts (5 Geography, 5 History, 5 Science) and asks each fact
in all 3 supported languages (English, Hindi, Gujarati) -> 45 questions total.

This allows meaningful cross-language comparison: retrieval quality,
groundedness, and hallucination rate can be compared for the SAME
underlying fact across languages, rather than comparing unrelated
question sets.

The knowledge base contains one matching document per fact per language
(45 documents) plus 9 distractor documents (3 facts x 3 languages) that are
topically related but do not directly answer any question. Distractors make
retrieval and similarity-based analysis non-trivial instead of a guaranteed
top-1 match.

All content below is original reference material written for this project.
The `source` field in the knowledge base honestly reflects this -- it does
NOT claim to be sourced from any specific external publication.

Run with:
    python -m src.build_seed_data
"""

import pandas as pd

from src import config

# ---------------------------------------------------------------------------
# QUESTIONS DATA
# ---------------------------------------------------------------------------
# Each entry represents one underlying fact, asked in three languages.
# fact_id groups the three language variants of the same question together,
# which downstream evaluation code uses for cross-language comparison.

QUESTIONS_RAW = [
    # ---------------- GEOGRAPHY ----------------
    {
        "fact_id": "GEO01",
        "category": "Geography",
        "variants": {
            "English": ("What is the capital of India?", "New Delhi"),
            "Hindi": ("भारत की राजधानी क्या है?", "नई दिल्ली"),
            "Gujarati": ("ભારતની રાજધાની શું છે?", "નવી દિલ્હી"),
        },
    },
    {
        "fact_id": "GEO02",
        "category": "Geography",
        "variants": {
            "English": (
                "Which river is generally considered the longest in the world?",
                "The Nile River",
            ),
            "Hindi": (
                "विश्व की सबसे लंबी नदी कौन सी मानी जाती है?",
                "नील नदी",
            ),
            "Gujarati": (
                "વિશ્વની સૌથી લાંબી નદી કઈ ગણાય છે?",
                "નાઇલ નદી",
            ),
        },
    },
    {
        "fact_id": "GEO03",
        "category": "Geography",
        "variants": {
            "English": (
                "Which mountain is the tallest above sea level?",
                "Mount Everest",
            ),
            "Hindi": (
                "समुद्र तल से सबसे ऊँचा पर्वत कौन सा है?",
                "माउंट एवरेस्ट",
            ),
            "Gujarati": (
                "સમુદ્ર સપાટીથી સૌથી ઊંચો પર્વત કયો છે?",
                "માઉન્ટ એવરેસ્ટ",
            ),
        },
    },
    {
        "fact_id": "GEO04",
        "category": "Geography",
        "variants": {
            "English": ("Which is the largest ocean on Earth?", "The Pacific Ocean"),
            "Hindi": ("पृथ्वी का सबसे बड़ा महासागर कौन सा है?", "प्रशांत महासागर"),
            "Gujarati": ("પૃથ્વીનો સૌથી મોટો મહાસાગર કયો છે?", "પ્રશાંત મહાસાગર"),
        },
    },
    {
        "fact_id": "GEO05",
        "category": "Geography",
        "variants": {
            "English": (
                "Which continent is the second largest by both area and population?",
                "Africa",
            ),
            "Hindi": (
                "क्षेत्रफल और जनसंख्या दोनों के हिसाब से दूसरा सबसे बड़ा महाद्वीप कौन सा है?",
                "अफ्रीका",
            ),
            "Gujarati": (
                "ક્ષેત્રફળ અને વસ્તી બંનેની દ્રષ્ટિએ બીજો સૌથી મોટો ખંડ કયો છે?",
                "આફ્રિકા",
            ),
        },
    },
    # ---------------- HISTORY ----------------
    {
        "fact_id": "HIS01",
        "category": "History",
        "variants": {
            "English": (
                "On what date did India gain independence from British rule?",
                "15 August 1947",
            ),
            "Hindi": (
                "भारत को ब्रिटिश शासन से किस तारीख को स्वतंत्रता मिली?",
                "15 अगस्त 1947",
            ),
            "Gujarati": (
                "ભારતને બ્રિટિશ શાસનથી કઈ તારીખે આઝાદી મળી?",
                "15 ઓગસ્ટ 1947",
            ),
        },
    },
    {
        "fact_id": "HIS02",
        "category": "History",
        "variants": {
            "English": ("Who built the Taj Mahal?", "Mughal emperor Shah Jahan"),
            "Hindi": ("ताजमहल किसने बनवाया था?", "मुगल सम्राट शाहजहाँ"),
            "Gujarati": ("તાજમહલ કોણે બંધાવ્યો હતો?", "મુઘલ સમ્રાટ શાહજહાં"),
        },
    },
    {
        "fact_id": "HIS03",
        "category": "History",
        "variants": {
            "English": ("In what year did World War II end?", "1945"),
            "Hindi": ("द्वितीय विश्व युद्ध किस वर्ष समाप्त हुआ?", "1945"),
            "Gujarati": ("બીજું વિશ્વ યુદ્ધ કયા વર્ષે સમાપ્ત થયું?", "1945"),
        },
    },
    {
        "fact_id": "HIS04",
        "category": "History",
        "variants": {
            "English": (
                "Who led India's non-violent independence movement?",
                "Mahatma Gandhi",
            ),
            "Hindi": (
                "भारत के अहिंसक स्वतंत्रता आंदोलन का नेतृत्व किसने किया?",
                "महात्मा गांधी",
            ),
            "Gujarati": (
                "ભારતના અહિંસક સ્વતંત્રતા આંદોલનનું નેતૃત્વ કોણે કર્યું?",
                "મહાત્મા ગાંધી",
            ),
        },
    },
    {
        "fact_id": "HIS05",
        "category": "History",
        "variants": {
            "English": (
                "Which is one of the world's oldest urban civilizations?",
                "The Indus Valley Civilization",
            ),
            "Hindi": (
                "विश्व की सबसे पुरानी शहरी सभ्यताओं में से एक कौन सी है?",
                "सिंधु घाटी सभ्यता",
            ),
            "Gujarati": (
                "વિશ્વની સૌથી જૂની શહેરી સંસ્કૃતિઓમાંની એક કઈ છે?",
                "સિંધુ ખીણની સંસ્કૃતિ",
            ),
        },
    },
    # ---------------- SCIENCE ----------------
    {
        "fact_id": "SCI01",
        "category": "Science",
        "variants": {
            "English": (
                "What are the two elements that make up water?",
                "Hydrogen and oxygen",
            ),
            "Hindi": ("पानी किन दो तत्वों से मिलकर बना है?", "हाइड्रोजन और ऑक्सीजन"),
            "Gujarati": (
                "પાણી કયા બે તત્વોથી બનેલું છે?",
                "હાઇડ્રોજન અને ઓક્સિજન",
            ),
        },
    },
    {
        "fact_id": "SCI02",
        "category": "Science",
        "variants": {
            "English": ("How many chambers does the human heart have?", "Four"),
            "Hindi": ("मानव हृदय में कितने कक्ष होते हैं?", "चार"),
            "Gujarati": ("માનવ હૃદયમાં કેટલા ખંડ (ચેમ્બર) હોય છે?", "ચાર"),
        },
    },
    {
        "fact_id": "SCI03",
        "category": "Science",
        "variants": {
            "English": ("Which planet is third from the Sun?", "Earth"),
            "Hindi": ("सूर्य से तीसरा ग्रह कौन सा है?", "पृथ्वी"),
            "Gujarati": ("સૂર્યથી ત્રીજો ગ્રહ કયો છે?", "પૃથ્વી"),
        },
    },
    {
        "fact_id": "SCI04",
        "category": "Science",
        "variants": {
            "English": (
                "What is the process by which plants convert sunlight into energy called?",
                "Photosynthesis",
            ),
            "Hindi": (
                "पौधों द्वारा सूर्य के प्रकाश को ऊर्जा में बदलने की प्रक्रिया को क्या कहते हैं?",
                "प्रकाश संश्लेषण",
            ),
            "Gujarati": (
                "છોડ સૂર્યપ્રકાશને ઊર્જામાં રૂપાંતરિત કરવાની પ્રક્રિયાને શું કહેવાય છે?",
                "પ્રકાશસંશ્લેષણ",
            ),
        },
    },
    {
        "fact_id": "SCI05",
        "category": "Science",
        "variants": {
            "English": ("What is the chemical symbol for gold?", "Au"),
            "Hindi": ("सोने का रासायनिक प्रतीक क्या है?", "Au (स्वर्ण)"),
            "Gujarati": ("સોનાનું રાસાયણિક પ્રતીક શું છે?", "Au"),
        },
    },
]

# ---------------------------------------------------------------------------
# KNOWLEDGE BASE DATA
# ---------------------------------------------------------------------------
# One matching document per fact per language (45 total), keyed by the same
# fact_id as the questions above so retrieval quality can be traced back to
# a specific fact/category/language combination during evaluation.

KB_MATCHING_RAW = [
    {
        "fact_id": "GEO01",
        "category": "Geography",
        "content": {
            "English": (
                "New Delhi is the capital of India. It is also the seat of "
                "all three branches of the Government of India."
            ),
            "Hindi": (
                "नई दिल्ली भारत की राजधानी है। यह भारत सरकार की तीनों शाखाओं "
                "की सीट भी है।"
            ),
            "Gujarati": (
                "નવી દિલ્હી ભારતની રાજધાની છે. તે ભારત સરકારની ત્રણેય "
                "શાખાઓની બેઠક પણ છે."
            ),
        },
    },
    {
        "fact_id": "GEO02",
        "category": "Geography",
        "content": {
            "English": (
                "The Nile River, flowing through northeastern Africa, is "
                "generally considered the longest river in the world at "
                "approximately 6,650 kilometers."
            ),
            "Hindi": (
                "नील नदी, जो उत्तर-पूर्वी अफ्रीका से होकर बहती है, को आमतौर पर "
                "विश्व की सबसे लंबी नदी माना जाता है, जिसकी लंबाई लगभग 6,650 "
                "किलोमीटर है।"
            ),
            "Gujarati": (
                "નાઇલ નદી, જે ઉત્તર-પૂર્વ આફ્રિકામાંથી વહે છે, તેને સામાન્ય "
                "રીતે વિશ્વની સૌથી લાંબી નદી ગણવામાં આવે છે, જેની લંબાઈ "
                "આશરે 6,650 કિલોમીટર છે."
            ),
        },
    },
    {
        "fact_id": "GEO03",
        "category": "Geography",
        "content": {
            "English": (
                "Mount Everest, located in the Himalayas on the border of "
                "Nepal and Tibet, is the tallest mountain above sea level "
                "at 8,849 meters."
            ),
            "Hindi": (
                "माउंट एवरेस्ट, जो नेपाल और तिब्बत की सीमा पर हिमालय में "
                "स्थित है, समुद्र तल से 8,849 मीटर की ऊंचाई के साथ सबसे "
                "ऊंचा पर्वत है।"
            ),
            "Gujarati": (
                "માઉન્ટ એવરેસ્ટ, જે નેપાળ અને તિબેટની સરહદ પર હિમાલયમાં "
                "આવેલો છે, તે સમુદ્ર સપાટીથી 8,849 મીટરની ઊંચાઈ સાથે સૌથી "
                "ઊંચો પર્વત છે."
            ),
        },
    },
    {
        "fact_id": "GEO04",
        "category": "Geography",
        "content": {
            "English": (
                "The Pacific Ocean is the largest and deepest of Earth's "
                "oceanic divisions, covering more area than all of Earth's "
                "land combined."
            ),
            "Hindi": (
                "प्रशांत महासागर पृथ्वी के महासागरीय विभाजनों में सबसे बड़ा "
                "और सबसे गहरा है, जो पृथ्वी की सभी भूमि से अधिक क्षेत्र को "
                "कवर करता है।"
            ),
            "Gujarati": (
                "પ્રશાંત મહાસાગર પૃથ્વીના મહાસાગરીય વિભાગોમાં સૌથી મોટો "
                "અને સૌથી ઊંડો છે, જે પૃથ્વીની તમામ જમીન કરતાં વધુ વિસ્તાર "
                "આવરી લે છે."
            ),
        },
    },
    {
        "fact_id": "GEO05",
        "category": "Geography",
        "content": {
            "English": (
                "Africa is the second largest continent in the world by "
                "both land area and population, after Asia."
            ),
            "Hindi": (
                "अफ्रीका एशिया के बाद क्षेत्रफल और जनसंख्या दोनों के "
                "हिसाब से विश्व का दूसरा सबसे बड़ा महाद्वीप है।"
            ),
            "Gujarati": (
                "આફ્રિકા એશિયા પછી ક્ષેત્રફળ અને વસ્તી બંનેની દ્રષ્ટિએ "
                "વિશ્વનો બીજો સૌથી મોટો ખંડ છે."
            ),
        },
    },
    {
        "fact_id": "HIS01",
        "category": "History",
        "content": {
            "English": (
                "India gained independence from British colonial rule on "
                "15 August 1947, ending nearly two centuries of British "
                "governance."
            ),
            "Hindi": (
                "भारत को 15 अगस्त 1947 को ब्रिटिश औपनिवेशिक शासन से "
                "स्वतंत्रता मिली, जिससे लगभग दो शताब्दियों के ब्रिटिश "
                "शासन का अंत हुआ।"
            ),
            "Gujarati": (
                "ભારતને 15 ઓગસ્ટ 1947ના રોજ બ્રિટિશ સંસ્થાનવાદી શાસનથી "
                "આઝાદી મળી, જેનાથી લગભગ બે સદીના બ્રિટિશ શાસનનો અંત આવ્યો."
            ),
        },
    },
    {
        "fact_id": "HIS02",
        "category": "History",
        "content": {
            "English": (
                "The Taj Mahal was commissioned by Mughal emperor Shah "
                "Jahan in memory of his wife Mumtaz Mahal and completed "
                "around 1653."
            ),
            "Hindi": (
                "ताजमहल का निर्माण मुगल सम्राट शाहजहाँ ने अपनी पत्नी "
                "मुमताज़ महल की याद में करवाया था और यह लगभग 1653 में "
                "पूरा हुआ।"
            ),
            "Gujarati": (
                "તાજમહલનું નિર્માણ મુઘલ સમ્રાટ શાહજહાંએ તેમની પત્ની "
                "મુમતાઝ મહલની યાદમાં કરાવ્યું હતું અને તે આશરે 1653માં "
                "પૂર્ણ થયું."
            ),
        },
    },
    {
        "fact_id": "HIS03",
        "category": "History",
        "content": {
            "English": (
                "World War II ended in 1945 with the surrender of Germany "
                "in May and Japan in September."
            ),
            "Hindi": (
                "द्वितीय विश्व युद्ध 1945 में समाप्त हुआ, जिसमें मई में "
                "जर्मनी और सितंबर में जापान ने आत्मसमर्पण किया।"
            ),
            "Gujarati": (
                "બીજું વિશ્વ યુદ્ધ 1945માં સમાપ્ત થયું, જેમાં મે મહિનામાં "
                "જર્મનીએ અને સપ્ટેમ્બરમાં જાપાને શરણાગતિ સ્વીકારી."
            ),
        },
    },
    {
        "fact_id": "HIS04",
        "category": "History",
        "content": {
            "English": (
                "Mahatma Gandhi led India's non-violent independence "
                "movement using strategies such as civil disobedience and "
                "non-cooperation."
            ),
            "Hindi": (
                "महात्मा गांधी ने सविनय अवज्ञा और असहयोग जैसी रणनीतियों "
                "का उपयोग करते हुए भारत के अहिंसक स्वतंत्रता आंदोलन का "
                "नेतृत्व किया।"
            ),
            "Gujarati": (
                "મહાત્મા ગાંધીએ સવિનય કાનૂનભંગ અને અસહકાર જેવી "
                "વ્યૂહરચનાઓનો ઉપયોગ કરીને ભારતના અહિંસક સ્વતંત્રતા "
                "આંદોલનનું નેતૃત્વ કર્યું."
            ),
        },
    },
    {
        "fact_id": "HIS05",
        "category": "History",
        "content": {
            "English": (
                "The Indus Valley Civilization, which flourished around "
                "2500 BCE, is one of the world's oldest urban "
                "civilizations, known for planned cities like Mohenjo-daro "
                "and Harappa."
            ),
            "Hindi": (
                "सिंधु घाटी सभ्यता, जो लगभग 2500 ईसा पूर्व फली-फूली, "
                "विश्व की सबसे पुरानी शहरी सभ्यताओं में से एक है, जो "
                "मोहनजोदड़ो और हड़प्पा जैसे नियोजित शहरों के लिए जानी "
                "जाती है।"
            ),
            "Gujarati": (
                "સિંધુ ખીણની સંસ્કૃતિ, જે આશરે 2500 ઈસા પૂર્વે વિકસી "
                "હતી, તે વિશ્વની સૌથી જૂની શહેરી સંસ્કૃતિઓમાંની એક છે, જે "
                "મોહેંજો-દડો અને હડપ્પા જેવા આયોજિત શહેરો માટે જાણીતી છે."
            ),
        },
    },
    {
        "fact_id": "SCI01",
        "category": "Science",
        "content": {
            "English": (
                "Water is a chemical compound made up of two hydrogen "
                "atoms and one oxygen atom, giving it the formula H2O."
            ),
            "Hindi": (
                "पानी एक रासायनिक यौगिक है जो दो हाइड्रोजन परमाणुओं और "
                "एक ऑक्सीजन परमाणु से मिलकर बना है, जिसका सूत्र H2O है।"
            ),
            "Gujarati": (
                "પાણી એક રાસાયણિક સંયોજન છે જે બે હાઇડ્રોજન અણુઓ અને એક "
                "ઓક્સિજન અણુથી બનેલું છે, જેનું સૂત્ર H2O છે."
            ),
        },
    },
    {
        "fact_id": "SCI02",
        "category": "Science",
        "content": {
            "English": (
                "The human heart has four chambers: two atria and two "
                "ventricles, which work together to pump blood through "
                "the body."
            ),
            "Hindi": (
                "मानव हृदय में चार कक्ष होते हैं: दो अलिंद (एट्रिया) और "
                "दो निलय (वेंट्रिकल्स), जो मिलकर शरीर में रक्त पंप करते "
                "हैं।"
            ),
            "Gujarati": (
                "માનવ હૃદયમાં ચાર ખંડ (ચેમ્બર) હોય છે: બે કર્ણક "
                "(એટ્રિયા) અને બે ક્ષેપક (વેન્ટ્રિકલ્સ), જે સાથે મળીને "
                "શરીરમાં લોહી પંપ કરે છે."
            ),
        },
    },
    {
        "fact_id": "SCI03",
        "category": "Science",
        "content": {
            "English": (
                "Earth is the third planet from the Sun in the solar "
                "system, situated between Venus and Mars."
            ),
            "Hindi": (
                "पृथ्वी सौर मंडल में सूर्य से तीसरा ग्रह है, जो शुक्र और "
                "मंगल के बीच स्थित है।"
            ),
            "Gujarati": (
                "પૃથ્વી સૂર્યમંડળમાં સૂર્યથી ત્રીજો ગ્રહ છે, જે શુક્ર "
                "અને મંગળ વચ્ચે સ્થિત છે."
            ),
        },
    },
    {
        "fact_id": "SCI04",
        "category": "Science",
        "content": {
            "English": (
                "Photosynthesis is the process by which green plants use "
                "sunlight, water, and carbon dioxide to produce glucose "
                "and oxygen."
            ),
            "Hindi": (
                "प्रकाश संश्लेषण वह प्रक्रिया है जिसके द्वारा हरे पौधे "
                "सूर्य के प्रकाश, पानी और कार्बन डाइऑक्साइड का उपयोग "
                "करके ग्लूकोज और ऑक्सीजन का उत्पादन करते हैं।"
            ),
            "Gujarati": (
                "પ્રકાશસંશ્લેષણ એ પ્રક્રિયા છે જેના દ્વારા લીલા છોડ "
                "સૂર્યપ્રકાશ, પાણી અને કાર્બન ડાયોક્સાઇડનો ઉપયોગ કરીને "
                "ગ્લુકોઝ અને ઓક્સિજન ઉત્પન્ન કરે છે."
            ),
        },
    },
    {
        "fact_id": "SCI05",
        "category": "Science",
        "content": {
            "English": (
                "The chemical symbol for gold is Au, derived from the "
                "Latin word 'aurum', and its atomic number is 79."
            ),
            "Hindi": (
                "सोने का रासायनिक प्रतीक Au है, जो लैटिन शब्द 'ऑरम' "
                "(aurum) से लिया गया है, और इसकी परमाणु संख्या 79 है।"
            ),
            "Gujarati": (
                "સોનાનું રાસાયણિક પ્રતીક Au છે, જે લેટિન શબ્દ 'ઓરમ' "
                "(aurum) પરથી ઉતરી આવ્યું છે, અને તેનો પરમાણુ ક્રમાંક "
                "79 છે."
            ),
        },
    },
]

# Distractor documents: topically related to a category but do NOT directly
# answer any question above. These make retrieval and similarity scoring
# non-trivial rather than a guaranteed top-1 match every time.
KB_DISTRACTORS_RAW = [
    {
        "distractor_id": "DIST01",
        "category": "Geography",
        "content": {
            "English": (
                "The Amazon rainforest is the largest tropical rainforest "
                "in the world, located primarily in Brazil."
            ),
            "Hindi": (
                "अमेज़न वर्षावन दुनिया का सबसे बड़ा उष्णकटिबंधीय वर्षावन "
                "है, जो मुख्य रूप से ब्राज़ील में स्थित है।"
            ),
            "Gujarati": (
                "એમેઝોન વરસાદી જંગલ વિશ્વનું સૌથી મોટું ઉષ્ણકટિબંધીય "
                "વરસાદી જંગલ છે, જે મુખ્યત્વે બ્રાઝિલમાં આવેલું છે."
            ),
        },
    },
    {
        "distractor_id": "DIST02",
        "category": "History",
        "content": {
            "English": (
                "The Roman Empire was one of the largest empires in "
                "ancient history, centered on the Mediterranean Sea."
            ),
            "Hindi": (
                "रोमन साम्राज्य प्राचीन इतिहास के सबसे बड़े साम्राज्यों "
                "में से एक था, जो भूमध्य सागर के आसपास केंद्रित था।"
            ),
            "Gujarati": (
                "રોમન સામ્રાજ્ય પ્રાચીન ઇતિહાસના સૌથી મોટા "
                "સામ્રાજ્યોમાંનું એક હતું, જે ભૂમધ્ય સમુદ્રની આસપાસ "
                "કેન્દ્રિત હતું."
            ),
        },
    },
    {
        "distractor_id": "DIST03",
        "category": "Science",
        "content": {
            "English": (
                "The speed of light in a vacuum is approximately 299,792 "
                "kilometers per second."
            ),
            "Hindi": (
                "निर्वात में प्रकाश की गति लगभग 299,792 किलोमीटर प्रति "
                "सेकंड है।"
            ),
            "Gujarati": (
                "શૂન્યાવકાશમાં પ્રકાશની ઝડપ આશરે 299,792 કિલોમીટર "
                "પ્રતિ સેકન્ડ છે."
            ),
        },
    },
]

SOURCE_LABEL = "Project knowledge base (original reference content compiled for this project)"


def build_questions_dataframe() -> pd.DataFrame:
    """Flattens QUESTIONS_RAW into one row per (fact_id, language)."""
    rows = []
    for fact in QUESTIONS_RAW:
        for language, (question_text, reference_answer) in fact["variants"].items():
            lang_code = config.LANGUAGE_CODES[language]
            rows.append(
                {
                    "question_id": f"{fact['fact_id']}_{lang_code}",
                    "category": fact["category"],
                    "language": language,
                    "question": question_text,
                    "reference_answer": reference_answer,
                }
            )
    df = pd.DataFrame(rows)
    return df


def build_knowledge_base_dataframe() -> pd.DataFrame:
    """Flattens matching + distractor KB entries into one row per document."""
    rows = []
    doc_counter = 1

    for fact in KB_MATCHING_RAW:
        for language, content_text in fact["content"].items():
            rows.append(
                {
                    "document_id": f"KB{doc_counter:03d}",
                    "category": fact["category"],
                    "language": language,
                    "content": content_text,
                    "source": SOURCE_LABEL,
                    "linked_fact_id": fact["fact_id"],
                    "is_distractor": False,
                }
            )
            doc_counter += 1

    for distractor in KB_DISTRACTORS_RAW:
        for language, content_text in distractor["content"].items():
            rows.append(
                {
                    "document_id": f"KB{doc_counter:03d}",
                    "category": distractor["category"],
                    "language": language,
                    "content": content_text,
                    "source": SOURCE_LABEL,
                    "linked_fact_id": distractor["distractor_id"],
                    "is_distractor": True,
                }
            )
            doc_counter += 1

    df = pd.DataFrame(rows)
    return df


def main() -> None:
    questions_df = build_questions_dataframe()
    kb_df = build_knowledge_base_dataframe()

    # utf-8-sig ensures Hindi/Gujarati text displays correctly if the CSV
    # is opened directly in Excel on Windows.
    questions_df.to_csv(config.QUESTIONS_CSV, index=False, encoding="utf-8-sig")
    kb_df.to_csv(config.KNOWLEDGE_BASE_CSV, index=False, encoding="utf-8-sig")

    print(f"Wrote {len(questions_df)} questions to: {config.QUESTIONS_CSV}")
    print(f"Wrote {len(kb_df)} knowledge base documents to: {config.KNOWLEDGE_BASE_CSV}")
    print()
    print("Questions per language:")
    print(questions_df["language"].value_counts().to_string())
    print()
    print("Knowledge base documents per language:")
    print(kb_df["language"].value_counts().to_string())
    print()
    print(f"Distractor documents: {int(kb_df['is_distractor'].sum())}")


if __name__ == "__main__":
    main()