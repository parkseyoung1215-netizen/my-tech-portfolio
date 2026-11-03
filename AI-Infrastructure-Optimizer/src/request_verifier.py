"""Runtime arithmetic verification from the request, never evaluation labels.

This is an explicit calculator-backed verifier for the supported arithmetic domain.
It never supplies a replacement answer or corrected numeric feedback to a model.
"""

import re

if __package__:
    from .answer_validator import extract_answer
else:
    from answer_validator import extract_answer


def verify_response(question, response):
    predicted = extract_answer(response)
    if predicted is None:
        return {"valid": False, "reason": "extraction_failed", "predicted": None}
    if not isinstance(question, str) or len(question) > 128:
        return {"valid": False, "reason": "unsupported_request", "predicted": predicted}
    text = question.strip()
    match = re.fullmatch(r"([0-9]{1,3})\s*([+*])\s*([0-9]{1,3})", text)
    if match:
        a, operation, b = match.groups()
        value = int(a) + int(b) if operation == "+" else int(a) * int(b)
    else:
        match = re.fullmatch(
            r"\(\s*([0-9]{1,3})\s*\*\s*([0-9]{1,3})\s*\)\s*-\s*([0-9]{1,3})", text
        )
        if match is None:
            return {"valid": False, "reason": "unsupported_request", "predicted": predicted}
        a, b, c = map(int, match.groups())
        value = a * b - c
    return {"valid": predicted == value,
            "reason": "verified" if predicted == value else "arithmetic_mismatch",
            "predicted": predicted}
