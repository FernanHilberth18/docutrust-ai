import re

SUSPICIOUS_PATTERNS = [
    r"ignore (all |any )?(previous|prior) instructions",
    r"system prompt",
    r"developer message",
    r"reveal (the )?(secret|password|token)",
    r"ignora (todas )?las instrucciones",
    r"mensaje (del )?sistema",
    r"revela (el )?(secreto|token|password|contraseña)",
]


def contains_untrusted_instruction(text: str) -> bool:
    lowered = text.lower()
    return any(re.search(pattern, lowered) for pattern in SUSPICIOUS_PATTERNS)
