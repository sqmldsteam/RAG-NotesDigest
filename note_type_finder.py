# Cell 1: Import dependencies
import re

# Cell 2: Rule-based weighted keyword classification
NOTE_TYPE_KEYWORDS = {
    "History & Physical": [
        "history & physical", "h&p", "admit date", "chief complaint", "past medical history"
    ],
    "Discharge Summary": [
        "discharge summary", "discharge medications", "discharge instructions", "discharge disposition"
    ],
    "Consultation Note": [
        "consultation note", "requested consultation", "consulted by", "recommendations"
    ],
    "Progress Note": [
        "progress note", "soap", "subjective", "objective", "assessment", "plan", "daily update"
    ],
    "Procedure Note": [
        "procedure performed", "technique", "findings", "complications", "post-procedure"
    ],
    "ED Note": [
        "emergency department", "ed course", "triage", "arrival", "ed medications",
        "brought in", "initial assessment", "vitals", "admitted from ed", "ed physician"
    ]
}

# Cell 3: Function to classify note type
def classify_note_type_weighted(note_text):
    note_text_lower = note_text.lower()
    scores = {note_type: 0 for note_type in NOTE_TYPE_KEYWORDS.keys()}

    # Count keyword matches for each note type
    for note_type, keywords in NOTE_TYPE_KEYWORDS.items():
        for kw in keywords:
            if kw.lower() in note_text_lower:
                scores[note_type] += 1

    # Get note type with highest score
    predicted_note_type = max(scores, key=scores.get)
    
    # If no keywords match, fallback to generic
    if scores[predicted_note_type] == 0:
        predicted_note_type = "Generic Medical Note"
    
    return predicted_note_type, scores
