import time
import pandas as pd
from openai import OpenAI

client = OpenAI()
MODEL = "gpt-4o-mini"

PROMPTS = [
    "Write a product description for a water bottle in exactly two sentences.",
    'Summarize the benefits of sleep in exactly three bullet points. Do not use the word "health".',
    "Write a tweet announcing a new coffee shop. Keep it under 20 words and include no hashtags.",
    "Explain what a database index is in one sentence of at most 25 words.",
    'Write a polite email declining a meeting. Do not use the word "unfortunately".',
    "List five fruits in alphabetical order, separated by semicolons, with no other text.",
    "Describe the color blue to a child in exactly four sentences.",
    'Write a haiku about winter and do not use the word "snow".',
    "Give three tips for saving money. Each tip must be under ten words.",
    'Translate "Where is the train station?" into French and Spanish. Give only the two translations, one per line.',
    "Write a two-sentence horror story set in a library.",
    "Explain photosynthesis in exactly 40 words.",
    "Write a slogan for a bicycle brand in all capital letters and with no punctuation.",
    "Name three planets and do not mention Earth, Mars, or Jupiter.",
    "Write a thank-you note to a teacher in under 30 words.",
    'Describe a sunset in one sentence without using the words "red", "orange" or "gold".',
    "Give a numbered list of four steps to boil an egg. Each step must start with a verb.",
    "Write a four-line poem about a cat where every line begins with the letter C.",
    "Explain gravity to a ten-year-old in at most three sentences.",
    'Write a job title and a one-sentence description for a data analyst role. Do not use the word "data" in the description.',
    "Suggest a name for a bakery and explain it in exactly one sentence.",
    "Write a short apology message to a customer for a late delivery. Mention the order number 4471 and offer a 10 percent discount.",
    "List the days of the week in reverse order starting from Sunday, separated by commas, with no other text.",
    "Write a motivational quote with exactly seven words.",
    "Describe how to tie a shoe in exactly five numbered steps.",
    "Write a two-line rhyming couplet about the rain.",
    "Give an example of a metaphor and then an example of a simile. Label each.",
    "Write a welcome message for a new employee named Priya. Keep it to exactly two sentences and mention the Tuesday orientation.",
    'Write a product review of headphones in the first person without using the word "sound".',
    'Explain what inflation is in under 50 words and without using the word "prices".',
    "Write a text message inviting a friend to dinner on Friday at 7 pm. Include the restaurant name Luigi's.",
    'Give three synonyms for "happy" in a single line separated by slashes.',
    "Write a short bio for a photographer named Lena in the third person in exactly three sentences.",
    "Describe the water cycle using exactly four bullet points.",
    'Write a question I could ask in a job interview. Do not use the word "weakness".',
    "Write a sentence that contains every vowel at least once and is under 15 words.",
    'Rewrite this sentence in a formal tone: "Hey, can you send me that file asap?" Reply with only the rewritten sentence.',
    'Write a fun fact about octopuses in exactly one sentence and do not use the word "brain".',
    "Write a reminder message about a dentist appointment on Monday at 9 am. Keep it under 15 words.",
    "List four renewable energy sources in a bulleted list. Do not include solar.",
    'Write a one-paragraph description of autumn that includes the words "leaves" and "wind" but not "cold".',
    "Explain what an API is in exactly two sentences.",
    "Give two pros and two cons of remote work in a four-line list where each line starts with Pro or Con.",
    "Write a headline for a news story about a city opening a new park. Use at most eight words.",
    "Describe your ideal weekend in the second person in exactly three sentences.",
    'Write a riddle whose answer is "clock". Do not include the word "clock" in the riddle.',
    "Give three questions to ask a new neighbor. Number them and keep each under eight words.",
    "Write the opening line of a mystery novel that includes a lighthouse and a missing key.",
    'Explain recursion in one sentence that includes the word "itself".',
    "Write a short goodbye message to a coworker who is leaving. Keep it to two sentences and do not use exclamation marks.",
]

SUMMARY_INSTRUCTION = (
    "Describe in one sentence what the following text is and how it is written "
    "(its length, format, tone and notable content). Do not judge whether it is good.\n\nTEXT:\n"
)

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
for i, p in enumerate(PROMPTS, start=1):
    out = ask(p)
    summ = ask(SUMMARY_INSTRUCTION + out)
    rows.append({"id": i, "intent": p, "output": out, "summary": summ})
    print(f"done {i}/{len(PROMPTS)}")

df = pd.DataFrame(rows)
df.to_csv("results_v2/real_outputs.csv", index=False)
lab = df[["id", "intent", "output"]].copy()
lab["label"] = ""
lab.to_csv("real_to_label.csv", index=False)
print("saved results_v2/real_outputs.csv and real_to_label.csv")
