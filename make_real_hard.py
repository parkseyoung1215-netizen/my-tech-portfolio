import re
import string
import time
import pandas as pd
from openai import OpenAI

client = OpenAI()
MODEL = "gpt-4o-mini"
SUFFIX = " Reply with only the requested text."

TASKS = [
    ("Write exactly 25 words about the moon.", [("words_eq", 25)]),
    ("Write one sentence of exactly 12 words about coffee.", [("words_eq", 12), ("sent_eq", 1)]),
    ("Write exactly three sentences about rain. Each sentence must be exactly 6 words long.", [("sent_eq", 3), ("each_sent_words", 6)]),
    ("Write a sentence about a dog without using the letter e.", [("no_letter", "e"), ("sent_eq", 1)]),
    ("Write a two-sentence description of a train in entirely lowercase letters.", [("sent_eq", 2), ("lower", None)]),
    ("Describe the ocean in exactly 30 words without using the word blue.", [("words_eq", 30), ("forbid", "blue")]),
    ("Write a sentence about winter that has no commas and is at most 15 words.", [("no_comma", None), ("words_le", 15), ("sent_eq", 1)]),
    ("List exactly five animals, one per line, each line starting with a dash.", [("dash_lines_eq", 5)]),
    ("Write a three-sentence story about a robot. Do not use the words robot or machine.", [("sent_eq", 3), ("forbid", "robot"), ("forbid", "machine")]),
    ("Write a sentence of exactly 10 words that ends with the word tomorrow.", [("words_eq", 10), ("ends_with", "tomorrow")]),
    ("Write a sentence about pizza in all capital letters with at most 8 words.", [("upper", None), ("words_le", 8)]),
    ("Write exactly 50 words about the history of the bicycle.", [("words_eq", 50)]),
    ("Write a four-sentence paragraph about trees. Do not use the letter z or the word green.", [("sent_eq", 4), ("no_letter", "z"), ("forbid", "green")]),
    ("Write one sentence about music that starts with the word Imagine and contains exactly 9 words.", [("starts_with", "imagine"), ("words_eq", 9)]),
    ("Write exactly two sentences about the sun without using the word hot.", [("sent_eq", 2), ("forbid", "hot")]),
    ("Write a sentence about a cat that contains the words purple and window and no more than 14 words.", [("require", "purple"), ("require", "window"), ("words_le", 14)]),
    ("Write a sentence of exactly 7 words without using the letter a.", [("words_eq", 7), ("no_letter", "a")]),
    ("List exactly three colors as three lines starting with a dash and with no commas anywhere.", [("dash_lines_eq", 3), ("no_comma", None)]),
    ("Write a five-sentence paragraph about school. Every sentence must have exactly 5 words.", [("sent_eq", 5), ("each_sent_words", 5)]),
    ("Write a tweet about Monday with fewer than 12 words and no punctuation.", [("words_lt", 12), ("no_punct", None)]),
    ("Describe a mountain in exactly 20 words and in entirely lowercase letters.", [("words_eq", 20), ("lower", None)]),
    ("Write a sentence about the sky that contains no vowels except the letter o.", [("only_vowel_o", None), ("sent_eq", 1)]),
    ("Write exactly four short sentences about a garden. Do not use the word flower.", [("sent_eq", 4), ("forbid", "flower")]),
    ("Write a two-line poem about the moon with exactly 6 words per line.", [("lines_eq", 2), ("each_line_words", 6)]),
    ("Write a sentence about bread that ends with the word oven and has exactly 11 words.", [("ends_with", "oven"), ("words_eq", 11)]),
    ("Write a sentence about a lake using fewer than 10 words and the word calm.", [("words_lt", 10), ("require", "calm")]),
    ("Write a paragraph about birds of exactly 35 words.", [("words_eq", 35)]),
    ("Write three lines about spring where each line starts with the word The.", [("lines_eq", 3), ("each_line_starts", "the")]),
    ("Write one sentence about space without using the letters s or t.", [("no_letter", "s"), ("no_letter", "t"), ("sent_eq", 1)]),
    ("Write exactly 15 words about friendship with no commas.", [("words_eq", 15), ("no_comma", None)]),
    ("List exactly six words that rhyme with cat, one per line with no other text.", [("lines_eq", 6), ("each_line_words", 1)]),
    ("Write a two-sentence summary of how rain forms in all capital letters without using the word WATER.", [("sent_eq", 2), ("upper", None), ("forbid", "water")]),
    ("Write a 3-bullet list of study tips; each bullet starts with a dash and has at most 6 words.", [("dash_lines_eq", 3), ("each_line_words_le", 6)]),
    ("Write a sentence with exactly 13 words about a river and do not use the word water.", [("words_eq", 13), ("forbid", "water")]),
    ("Write two sentences about chocolate. The first must have exactly 4 words and the second exactly 9 words.", [("sent_words_list", [4, 9])]),
    ("Write one sentence about a library without using the letter e or the letter a.", [("no_letter", "e"), ("no_letter", "a")]),
    ("Write a motivational sentence of exactly 8 words with no commas and ending with the word forward.", [("words_eq", 8), ("no_comma", None), ("ends_with", "forward")]),
    ("Write exactly 45 words about football.", [("words_eq", 45)]),
    ("Write a sentence about a bridge in entirely lowercase letters with exactly 10 words and no periods.", [("lower", None), ("words_eq", 10), ("no_period", None)]),
    ("Write exactly three sentences about the internet. Do not use the words information or computer.", [("sent_eq", 3), ("forbid", "information"), ("forbid", "computer")]),
]

SUMMARY_INSTRUCTION = (
    "Describe in one sentence what the following text is and how it is written "
    "(its length, format, tone and notable content). Do not judge whether it is good.\n\nTEXT:\n"
)

def words(t):
    return re.findall(r"[A-Za-z0-9]+(?:['\u2019-][A-Za-z0-9]+)*", t)

def sentences(t):
    parts = re.split(r"(?<=[.!?])\s+", t.strip())
    return [p for p in parts if p.strip()]

def lines(t):
    return [l.strip() for l in t.splitlines() if l.strip()]

def strip_bullet(l):
    return re.sub(r"^[-\u2022*]\s*", "", l)

def check(text, rule, val):
    low = text.lower()
    if rule == "words_eq":
        return len(words(text)) == val
    if rule == "words_le":
        return len(words(text)) <= val
    if rule == "words_lt":
        return len(words(text)) < val
    if rule == "sent_eq":
        return len(sentences(text)) == val
    if rule == "each_sent_words":
        return all(len(words(s)) == val for s in sentences(text))
    if rule == "sent_words_list":
        s = sentences(text)
        return len(s) == len(val) and all(len(words(x)) == n for x, n in zip(s, val))
    if rule == "no_letter":
        return val not in low
    if rule == "lower":
        return text == text.lower()
    if rule == "upper":
        return text == text.upper()
    if rule == "forbid":
        return re.search(r"\b" + re.escape(val) + r"\b", text, re.I) is None
    if rule == "require":
        return re.search(r"\b" + re.escape(val) + r"\b", text, re.I) is not None
    if rule == "no_comma":
        return "," not in text
    if rule == "no_period":
        return "." not in text
    if rule == "no_punct":
        return not any(c in string.punctuation for c in text)
    if rule == "dash_lines_eq":
        return sum(1 for l in lines(text) if l.startswith("-")) == val
    if rule == "lines_eq":
        return len(lines(text)) == val
    if rule == "each_line_words":
        return all(len(words(strip_bullet(l))) == val for l in lines(text))
    if rule == "each_line_words_le":
        return all(len(words(strip_bullet(l))) <= val for l in lines(text))
    if rule == "each_line_starts":
        return all(words(l) and words(l)[0].lower() == val for l in lines(text))
    if rule == "starts_with":
        w = words(text)
        return bool(w) and w[0].lower() == val
    if rule == "ends_with":
        w = words(text)
        return bool(w) and w[-1].lower() == val
    if rule == "only_vowel_o":
        return set(c for c in low if c in "aeiou") <= {"o"}
    raise ValueError(rule)

def ask(prompt):
    for attempt in range(3):
        try:
            r = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=1.0,
            )
            return r.choices[0].message.content.strip()
        except Exception as e:
            print("retry after error:", e)
            time.sleep(3)
    return ""

rows = []
for i, (intent, rules) in enumerate(TASKS, start=1):
    out = ask(intent + SUFFIX)
    summ = ask(SUMMARY_INSTRUCTION + out)
    failed = [r for r, v in rules if not check(out, r, v)] if out else ["empty"]
    rows.append({"id": i, "intent": intent, "output": out, "summary": summ,
                 "label": int(len(failed) == 0), "failed_rules": ";".join(failed)})
    print(f"done {i}/{len(TASKS)}")

df = pd.DataFrame(rows)
df.to_csv("results_v2/real_hard.csv", index=False)
print("label counts (1 = followed all rules, 0 = broke at least one):")
print(df["label"].value_counts())
print("saved results_v2/real_hard.csv")
