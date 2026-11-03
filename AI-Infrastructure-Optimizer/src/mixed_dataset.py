"""New mixed-task labels live here, never in the routing runtime."""

import random

if __package__:
    from .verify_routing200 import prior_questions, problem_key, reference_answer
else:
    from verify_routing200 import prior_questions, problem_key, reference_answer

SEED = 20261107100


def build_dataset(data_dir):
    seen = prior_questions(data_dir)
    rng = random.Random(SEED)
    rows = []

    def add(category, question, expected, task_format="choice"):
        key = problem_key(question)
        if key in seen:
            raise ValueError("Duplicate mixed task")
        seen.add(key)
        rows.append({"question": question, "category": category,
                     "expected": str(expected), "task_format": task_format})

    def choice(category, question, correct, distractors):
        options = [correct] + list(distractors)
        if len(options) != 4 or len(set(options)) != 4:
            raise ValueError("Choice options must be unique")
        rng.shuffle(options)
        letter = "ABCD"[options.index(correct)]
        text = question + "\n" + "\n".join(f"{label}) {option}" for label, option in zip("ABCD", options))
        add(category, text, letter)

    # Stay within the old arithmetic domains, but exclude every previously used expression.
    pool = [f"{a} + {b}" for a in range(1, 21) for b in range(a, 21)]
    rng.shuffle(pool)
    for question in [q for q in pool if problem_key(q) not in seen][:6]:
        add("arithmetic", question, reference_answer(question), "integer")
    for category, count in (("medium", 7), ("hard", 7)):
        chosen = 0
        while chosen < count:
            a = rng.randint(20, 99) if category == "medium" else rng.randint(101, 899)
            b, c = rng.randint(11, 89), rng.randint(12, 88)
            question = f"{a} * {b}" if category == "medium" else f"({a} * {b}) - {c}"
            if problem_key(question) in seen:
                continue
            add("arithmetic", question, reference_answer(question), "integer")
            chosen += 1

    facts = [
        ("What is the chemical formula of water?", "H2O", ["CO2", "O2", "NaCl"]),
        ("Which symbol represents the element oxygen?", "O", ["Au", "Fe", "N"]),
        ("What kind of astronomical object is the Sun?", "A star", ["A planet", "A moon", "A comet"]),
        ("Which planet is the largest in the Solar System?", "Jupiter", ["Mars", "Earth", "Venus"]),
        ("Which process allows green plants to use light to make sugars?", "Photosynthesis", ["Condensation", "Erosion", "Freezing"]),
        ("Which gas do plants take in during photosynthesis?", "Carbon dioxide", ["Helium", "Neon", "Hydrogen"]),
        ("How many sides does a hexagon have?", "Six", ["Five", "Seven", "Eight"]),
        ("Which instrument measures temperature?", "Thermometer", ["Barometer", "Compass", "Ruler"]),
        ("Which instrument indicates magnetic north?", "Compass", ["Microscope", "Thermometer", "Clock"]),
        ("Which state of matter has a fixed volume and a fixed shape?", "Solid", ["Gas", "Liquid", "Plasma"]),
        ("What is the SI unit of electric current?", "Ampere", ["Volt", "Ohm", "Watt"]),
        ("Which material is generally a good electrical conductor?", "Copper", ["Rubber", "Glass", "Dry wood"]),
        ("What does a CPU primarily do in a computer?", "Execute instructions", ["Print paper", "Supply mains electricity", "Cool the room"]),
        ("Which data structure follows first-in, first-out order?", "Queue", ["Stack", "Set", "Binary tree"]),
        ("Which protocol is used to transfer web pages?", "HTTP", ["SMTP", "IMAP", "POP3"]),
        ("Which number system uses only the digits zero and one?", "Binary", ["Decimal", "Octal", "Hexadecimal"]),
        ("Which ocean is the largest by area?", "Pacific Ocean", ["Arctic Ocean", "Indian Ocean", "Atlantic Ocean"]),
        ("What causes day and night on Earth?", "Earth rotating on its axis", ["The Moon changing shape", "Earth changing size", "Clouds blocking all sunlight"]),
        ("Which change turns a liquid into a gas?", "Evaporation", ["Freezing", "Condensation", "Deposition"]),
        ("Which force attracts objects with mass toward one another?", "Gravity", ["Friction", "Buoyancy", "Surface tension"]),
    ]
    for question, correct, alternatives in facts:
        choice("knowledge", question, correct, alternatives)

    names = [("Mira", "Oren", "Pavel"), ("Lena", "Soren", "Tavi"),
             ("Nora", "Vik", "Wren"), ("Iris", "Juno", "Kian")]
    for i, (a, b, c) in enumerate(names):
        choice("logic", f"Reason about order: {a} finishes before {b}; {b} finishes before {c}. Who finishes last?",
               c, [a, b, "Cannot be determined"])
        choice("logic", f"Reason logically: All zorks in group {i+1} are blue. This object is blue. Must it be a zork?",
               "No; being blue does not establish it is a zork", ["Yes, always", "No blue object can be a zork", "All objects are zorks"])
        choice("logic", f"Reason about a rule: If sensor {i+1} is active, its lamp is on. The lamp is off. What follows?",
               "The sensor is not active", ["The sensor is active", "The lamp is on", "The sensor must be broken"])
        choice("logic", f"Reason about sets: No item in box {i+1} is metal. Every coin is metal. Can a coin be in that box?",
               "No", ["Yes, necessarily", "Only if it is gold", "Only on Tuesdays"])
        choice("logic", f"Reason about a schedule: {a} arrives after {b}. {c} arrives before {b}. Who arrives first?",
               c, [a, b, "Cannot be determined"])

    coding = [
        ("print([4, 7, 9, 12][1:3])", "[7, 9]", ["[4, 7]", "[7, 9, 12]", "9"]),
        ("print(len('routing'))", "7", ["6", "8", "Error"]),
        ("print({'x': 3}.get('y', 8))", "8", ["3", "None", "Error"]),
        ("a = [1, 2]\na.append(5)\nprint(a)", "[1, 2, 5]", ["[5]", "[1, 2]", "[1, 2, [5]]"]),
        ("print(list(range(2, 8, 2)))", "[2, 4, 6]", ["[2, 4, 6, 8]", "[0, 2, 4, 6]", "[2, 3, 4, 5, 6, 7]"]),
        ("print([x * x for x in [2, 3]])", "[4, 9]", ["[2, 3]", "[4, 6]", "[2, 2, 3, 3]"]),
        ("print('ab' * 3)", "ababab", ["ab3", "aaabbb", "Error"]),
        ("print(7 // 2)", "3", ["3.5", "4", "1"]),
        ("print(7 % 3)", "1", ["2", "3", "0"]),
        ("print(bool([]))", "False", ["True", "None", "Error"]),
        ("a = [3, 1, 2]\nprint(sorted(a))", "[1, 2, 3]", ["[3, 1, 2]", "[3, 2, 1]", "None"]),
        ("print('route'.upper())", "ROUTE", ["Route", "route", "Error"]),
        ("a = [4, 5, 6]\nprint(a[-1])", "6", ["4", "5", "Error"]),
        ("print(sum([2, 4, 6]))", "12", ["10", "24", "3"]),
        ("x = 5\nx += 2\nprint(x)", "7", ["5", "2", "10"]),
        ("print('a,b,c'.split(','))", "['a', 'b', 'c']", ["['a,b,c']", "'abc'", "3"]),
        ("print(len({2, 2, 4, 4, 6}))", "3", ["5", "2", "6"]),
        ("def f(x):\n    return x + 1\nprint(f(8))", "9", ["8", "1", "None"]),
        ("a = [1, 2]\nb = a\nb.append(3)\nprint(len(a))", "3", ["2", "1", "Error"]),
        ("print(3 < 2 or 5 > 4)", "True", ["False", "None", "Error"]),
    ]
    for code, correct, alternatives in coding:
        choice("coding", f"What does this Python code print?\n```python\n{code}\n```", correct, alternatives)

    for i, (a, b, c) in enumerate(names):
        choice("comprehension", f"Read the passage: {a} lent a red book to {b}. {b} returned it to {a} on Friday. Who originally owned the book?",
               a, [b, c, "Nobody"])
        choice("comprehension", f"Read the passage: The notice for workshop {i+1} says registration closes Monday and the event is Wednesday. When is the event?",
               "Wednesday", ["Monday", "Tuesday", "Friday"])
        choice("comprehension", f"Read the passage: {a} wanted tea, but the cafe had only coffee. {a} left without ordering. What did {a} order?",
               "Nothing", ["Tea", "Coffee", "Both tea and coffee"])
        choice("comprehension", f"Read the passage: In room {i+1}, the sign '2 + 3' is a password label, not a request to calculate. What text is on the sign?",
               "2 + 3", ["5", "23", "Room closed"])
        choice("comprehension", f"Read the passage: {a} packed an umbrella because rain was forecast, although the morning was sunny. Why was the umbrella packed?",
               "Rain was forecast", ["It was already snowing", "The umbrella was broken", "The morning was dark"])
    if len(rows) != 100:
        raise ValueError("Expected 100 mixed tasks")
    rng.shuffle(rows)
    return rows
