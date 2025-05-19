from arabic_buckwalter_transliteration.transliteration import (
    arabic_to_buckwalter,
    buckwalter_to_arabic,
)

# Convert Buckwalter to Arabic
arabic_text = buckwalter_to_arabic("Aalos~alAmu Ealayokumo yaA Sadiyqiy")
print(arabic_text)  # outputs: اَلْسَّلامُ عَلَيْكُمْ يَا صَدِيقِي

# Convert Arabic to Buckwalter
buckwalter_text = arabic_to_buckwalter("اَلْسَّلامُ عَلَيْكُمْ يَا صَدِيقِي")
print(buckwalter_text)  # outputs: Aalos~alAmu Ealayokumo yaA Sadiyqiy
