"""Phase 11 - Safety policy: the detection rules behind the guardrails.

Each rule is a small regex/string check with a human-readable message. The
safety layer (guardrails.py) uses these to decide, in order:

  1. prompt injection      -> blocked immediately (highest priority)
  2. medical emergency     -> blocked, direct to emergency services
  3. personal / diagnosis  -> blocked, refer to a healthcare professional
  4. personal information  -> blocked, ask to remove the PII
  5. everything else       -> goes to the normal RAG pipeline (which itself
                              rejects out-of-scope questions via the relevance
                              gate from Phase 9)

The rules are deliberately conservative: when in doubt, a question is refused
with a clear message, never answered by guessing.
"""

import re

# ---------------------------------------------------------------------------
# User-facing messages
# ---------------------------------------------------------------------------
INJECTION_MESSAGE = (
    "I can only answer questions about HULIO using its Product Monograph. "
    "I can't follow instructions that are embedded inside a question."
)
EMERGENCY_MESSAGE = (
    "This looks like a medical emergency. Please call your local emergency "
    "services (or poison control) immediately."
)
PERSONAL_MESSAGE = (
    "HULIOMed is an educational assistant only. It cannot give personal "
    "medical advice, diagnose a condition, or recommend starting, stopping, "
    "or changing any treatment. For decisions about you or your family, "
    "please consult a qualified healthcare professional."
)
PII_MESSAGE = (
    "Please remove personal information (such as a phone number, email, "
    "PAN, Aadhaar, or bank/card number) from your question before "
    "continuing. HULIOMed does not process personal data."
)
UNSUPPORTED_MESSAGE = (
    "I could not verify this information in the Hulio Product Monograph. "
    "I can only answer based on what that document says about HULIO "
    "(adalimumab-fkjp). Please rephrase the question or ask about a topic "
    "covered in the monograph."
)

# ---------------------------------------------------------------------------
# 1. Prompt injection
# ---------------------------------------------------------------------------
INJECTION_PATTERNS = [
    r"\bignore\s+(?:all\s+)?(?:previous|prior|above|earlier|your)\s*"
    r"(?:instructions|prompt\w*|rules|guidelines|directions)\b",
    r"\bforget\s+(?:about\s+)?(?:your|all\s+the|the)\s*"
    r"(?:instructions|prompt\w*|rules|guidelines|system)\b",
    r"\bdisregard\b",
    r"\boverride\s+(?:your\b|the\b)?\s*(?:instructions|rules|prompt\w*)?",
    r"\bnew\s+instructions\b",
    r"\byou\s+are\s+now\b",
    r"\byour\s+system\s+prompt\b",
    r"\bprint\s+(?:out\s+)?(?:your|the\s+)?(?:instructions|system\s+prompt|"
    r"prompt)\b",
    r"\breveal\s+(?:your|the\s+)?(?:system|developer)?\s*(?:prompt|"
    r"instructions)\b",
    r"\bsystem\s+message\b",
    r"\bdeveloper\s+message\b",
    r"\bact\s+as\s+(?:a\s+|an\s+)?"
    r"(?:doctor|therapist|assistant|chatbot|helper|model)\b",
    r"\bjailbreak\b",
    r"\bfrom\s+now\s+on\b",
    r"\blet's\s+switch\b",
    r"\bno\s+rules\b",
]
INJECTION_RE = re.compile("|".join(INJECTION_PATTERNS), re.IGNORECASE)

# ---------------------------------------------------------------------------
# 2. Medical emergency (only when first person / urgent)
# ---------------------------------------------------------------------------
EMERGENCY_PATTERNS = [
    r"\bemergency\b",
    r"\bsuicid\w*",
    r"\boverdos\w*",
    r"\bpoison\w*",
    r"\banaphyla\w*",
    r"\bcan['']t\s+breathe\b",
    r"\bsevere\s+(?:allergic\s+)?reaction",
]
EMERGENCY_CONTEXT_PATTERNS = [
    r"\bi\b",
    r"\bmy\b",
    r"\bi'm\b",
    r"\bme\b",
    r"\bright\s+now\b",
    r"\bimmediately\b",
    r"\bnow\b",
]
EMERGENCY_RE = re.compile("|".join(EMERGENCY_PATTERNS), re.IGNORECASE)
EMERGENCY_CONTEXT_RE = re.compile(
    "|".join(EMERGENCY_CONTEXT_PATTERNS), re.IGNORECASE
)

# ---------------------------------------------------------------------------
# 3. Personal treatment / diagnosis requests
# ---------------------------------------------------------------------------
RELATIVES = (
    "mother|father|mom|dad|son|daughter|child|children|kid|grandmother|"
    "grandfather|grandma|grandpa|grandchild|grandchildren|wife|husband|"
    "spouse|partner|brother|sister|aunt|uncle|friend|patient|relative|"
    "family|parents?"
)
SYMPTOM_NOUNS = (
    "pain|psoriasis|arthritis|crohn|colitis|rash|fever|itch|symptom|"
    "disorder|disease|condition|diabetes|inflammation|inflammat|depression|"
    "anxiety|infection|tumor|tumour|cancer|headache|nausea|dizziness|"
    "swelling|injection.?site"
)
PERSONAL_PATTERNS = [
    rf"\b(?:should|can|could|may)\s+(?:i|me)\b",
    rf"\b(?:should|can|could|may)\s+my\b",
    rf"\bmy\s+(?:{RELATIVES})\b",
    rf"\bmy\s+(?:dose|dosage|injection|doses|treatment|doctor|condition|"
    rf"symptom|prescription)\b",
    rf"\b(?:diagnos\w*|prescrib\w*)\b",
    rf"\bi\s+(?:take|took|taking|was\s+prescribed|stopped|start|reduce)\b",
    rf"\bi['']m\s+taking\b",
    rf"\bi\s+have\s+\w*\s*(?:{SYMPTOM_NOUNS})\w*",
    rf"\b(?:i|my)\s+(?:have|has)\s+(?:a\s+)?(?:{SYMPTOM_NOUNS})",
]
PERSONAL_RE = re.compile("|".join(PERSONAL_PATTERNS), re.IGNORECASE)

# ---------------------------------------------------------------------------
# 4. Personal information (PII)
# ---------------------------------------------------------------------------
# Note: the monograph itself legitimately prints HULIO's public support phone
# numbers. Those are excluded here so they can still be quoted in answers.
PUBLIC_SUPPORT_NUMBERS = ("1-833-986-1468", "1-833-444-8546")

AADHAAR_RE = re.compile(r"\b\d{4}[ -]?\d{4}[ -]?\d{4}\b|\b\d{12}\b")
PAN_RE = re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b")
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"(?<!\d)(?:\+?\d[\s-]?){9,14}\d(?!\d)")
CARD_RE = re.compile(r"\b(?:\d[ -]?){13,16}\b")


def strip_public_numbers(text):
    for number in PUBLIC_SUPPORT_NUMBERS:
        text = text.replace(number, "")
    return text


def find_pii(text):
    """Return a list of PII kinds found in the text."""
    found = []
    if AADHAAR_RE.search(text):
        found.append("aadhaar")
    if PAN_RE.search(text):
        found.append("pan")
    if EMAIL_RE.search(text):
        found.append("email")
    if PHONE_RE.search(strip_public_numbers(text)):
        found.append("phone")
    if CARD_RE.search(strip_public_numbers(text)):
        found.append("card")
    return found


def is_prompt_injection(text):
    return bool(INJECTION_RE.search(text))


def is_personal_request(text):
    return bool(PERSONAL_RE.search(text))


def is_emergency(text):
    return bool(EMERGENCY_RE.search(text) and EMERGENCY_CONTEXT_RE.search(text))


def contains_disclaimer(text):
    """True if the text uses the official 'could not (be) verify' disclaimer."""
    low = text.lower()
    return "could not verify" in low or "could not be verified" in low