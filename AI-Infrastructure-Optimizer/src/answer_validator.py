
import re


def extract_answer(text):
    """
    Extract an integer from an explicit FINAL_ANSWER line.
    Return None if the answer is missing or ambiguous.
    """
    if not isinstance(text, str):
        return None

    # Count declarations, including malformed ones, to reject conflicting answers.
    declarations = [
        line for line in text.splitlines()
        if re.match(r"^[ \t]*FINAL_ANSWER\b", line, re.IGNORECASE)
    ]
    if len(declarations) != 1:
        return None

    match = re.fullmatch(
        r"[ \t]*FINAL_ANSWER[ \t]*:[ \t]*([+-]?[0-9]+)[ \t]*",
        declarations[0], re.IGNORECASE,
    )
    if match is None:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def validate_answer(response, expected):
    predicted = extract_answer(response)
    correct = predicted is not None and predicted == expected

    return {
        "predicted": predicted,
        "expected": expected,
        "correct": correct,
        "status": (
            "extraction_failed" if predicted is None
            else "correct" if correct else "incorrect"
        ),
    }


if __name__ == "__main__":
    examples = [
        ("4 + 8 = 12\nFINAL_ANSWER: 12", 12),
        ("4 + 8 = 12\nFINAL_ANSWER: 13", 12),
        ("The answer is 12", 12),
        ("FINAL_ANSWER: 12\nFINAL_ANSWER: 13", 12),
    ]

    for response, expected in examples:
        result = validate_answer(response, expected)
        print(result)
