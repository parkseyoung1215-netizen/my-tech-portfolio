"""Frozen calibration and new balanced 300-task evaluation; labels stay outside runtime."""
import json
import random
import re
from pathlib import Path

if __package__:
    from .verify_routing200 import ROOT, problem_key, reference_answer
else:
    from verify_routing200 import ROOT, problem_key, reference_answer

SEED = 20261108300
CALIBRATION_SOURCE = ROOT / "data/mixed_5f6833b806564466821eb9f81873848f_manifest.json"


def task_key(question):
    # Reordering choices does not make an old question new.
    stem = re.split(r"\nA\) ", question, maxsplit=1)[0]
    return problem_key(stem).lower()


def prior_keys(data_dir):
    import csv
    seen = set()
    def walk(obj):
        if isinstance(obj, dict):
            if isinstance(obj.get("question"), str): seen.add(task_key(obj["question"]))
            for v in obj.values(): walk(v)
        elif isinstance(obj, list):
            for v in obj: walk(v)
    for p in Path(data_dir).iterdir():
        if p.suffix == ".json": walk(json.loads(p.read_text()))
        elif p.suffix == ".csv":
            with p.open(newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f): walk(row)
    return seen


def build(data_dir, calibration_source=CALIBRATION_SOURCE):
    rng = random.Random(SEED)
    seen = prior_keys(data_dir)
    calibration = [t.copy() for t in json.loads(Path(calibration_source).read_text())["tasks"]
                   if "originally owned" not in t["question"]]
    evaluation = []
    def add(target, category, q, answer, kind="choice", reference_code=None):
        key = task_key(q)
        if key in seen: raise ValueError("Duplicate question stem or expression")
        seen.add(key)
        row = {"question": q, "category": category, "expected": str(answer), "task_format": kind}
        if reference_code is not None: row["reference_code"] = reference_code
        target.append(row)
    def choice(target, category, q, correct, wrong, code=None):
        opts = [str(correct)] + [str(x) for x in wrong]
        if len(opts) != 4 or len(set(opts)) != 4: raise ValueError("Invalid choices")
        rng.shuffle(opts)
        add(target, category, q+"\n"+"\n".join(f"{c}) {v}" for c,v in zip("ABCD",opts)),
            "ABCD"[opts.index(str(correct))], reference_code=code)
    # Replace the four ambiguous historical ownership items with explicit, separate calibration items.
    for i in range(4):
        choice(calibration,"comprehension",f"Read the passage: In calibration archive {i+101}, Neri owns a green notebook. Neri lends it to Sol. Who owns the notebook?",
               "Neri",["Sol","Nobody","Cannot be determined"])
    for operation in ("add","multiply","composite"):
        count = 0
        while count < 20:
            a,b,c = rng.randint(21,99),rng.randint(12,89),rng.randint(13,88)
            q = f"{a} + {b}" if operation=="add" else f"{a} * {b}" if operation=="multiply" else f"({a+200} * {b}) - {c}"
            if task_key(q) in seen: continue
            add(evaluation,"arithmetic",q,reference_answer(q),"integer"); count += 1
    facts = [
        ("Which data structure is last-in, first-out?","Stack",["Queue","Set","Graph"]),
        ("Which computer memory normally loses stored data when power is removed?","RAM",["ROM","SSD","Optical disc"]),
        ("Which storage is normally nonvolatile?","ROM",["CPU register","Ordinary RAM","CPU cache"]),
        ("How many bits are in a conventional byte?","8",["4","16","32"]),
        ("What is an algorithm?","A finite sequence of steps to solve a problem",["Only a processor","Only a cable","Only a display"]),
        ("What does a dictionary associate with each key?","A value",["Always a file","Always a thread","Always an image"]),
        ("Which pair are Boolean values?","True and False",["Red and Blue","Small and Large","Up and Down"]),
        ("When is Boolean AND true?","When both operands are true",["When both are false","Only when the left is false","Whenever either is true"]),
        ("When is Boolean OR false?","When both operands are false",["Whenever either is true","When both are true","Only when the left is true"]),
        ("What does Boolean NOT do?","Reverses a truth value",["Sorts a list","Duplicates a file","Adds two numbers"]),
        ("What does a row usually represent in a relational table?","One record",["A network cable","An operating system","A processor core"]),
        ("What must a primary key identify?","A unique row",["Every row with the same value","Only a table color","A screen position"]),
        ("What does a foreign key usually reference?","A key in another table",["A keyboard key","A display pixel","A CPU instruction address"]),
        ("What does ascending numeric sorting produce?","Smallest to largest",["Largest to smallest","Only duplicates","Random order"]),
        ("Which structure is normally used for depth-first search?","Stack",["FIFO queue","Printer spool","Spreadsheet cell"]),
        ("Which structure is normally used for breadth-first search?","FIFO queue",["LIFO stack","Single number","Text cursor"]),
        ("What stops a well-formed recursive computation?","A base case",["A larger monitor","A keyboard shortcut","A database column"]),
        ("Which Python sequence type is immutable?","Tuple",["List","Dictionary","Set"]),
        ("What property distinguishes a mathematical set from a multiset?","Duplicate elements are not counted separately",["Order is mandatory","All elements must be text","It contains exactly two elements"]),
        ("What is required before ordinary binary search on an array?","The array must be sorted",["Every element must be zero","The array must be random","The array must contain only letters"]),
        ("What is a prime integer?","An integer greater than one with exactly two positive divisors",["Any odd integer","Any negative integer","Any integer divisible by ten"]),
        ("Which property defines an even integer?","Divisibility by two",["Being positive","Being prime","Having three digits"]),
        ("How many degrees are in a right angle?","90",["45","180","360"]),
        ("What is the sum of interior angles of a Euclidean triangle?","180 degrees",["90 degrees","270 degrees","360 degrees"]),
        ("How many sides does an octagon have?","8",["6","7","9"]),
        ("How many sides does a pentagon have?","5",["4","6","8"]),
        ("Which formula gives circle area for radius r?","pi*r*r",["2*pi*r","pi+r","r+r"]),
        ("Which formula gives circle circumference for radius r?","2*pi*r",["pi*r*r","r*r*r","pi-r"]),
        ("What probability is assigned to a certain event?","1",["0","-1","2"]),
        ("What probability is assigned to an impossible event?","0",["1","-1","2"]),
        ("What is the median of an odd-length sorted numeric list?","The middle value",["Always the first value","Always the largest value","The total sum"]),
        ("What is the mode of a numeric data set?","A most frequent value",["The largest value","The smallest value","The total sum"]),
        ("How is the arithmetic mean calculated?","Sum divided by count",["Maximum minus minimum","Count divided by maximum","Product of all values"]),
        ("How does a circle diameter relate to its radius?","It is twice the radius",["It equals half the radius","It always equals one","It is unrelated"]),
        ("What is the perimeter of a square of side s?","4*s",["s*s","2*s","s/4"]),
        ("What is the area of a rectangle of width w and height h?","w*h",["2*(w+h)","w+h","w-h"]),
        ("How many vertices does a cube have?","8",["4","6","12"]),
        ("How many faces does a tetrahedron have?","4",["3","6","8"]),
        ("What is zero factorial?","1",["0","-1","Undefined"]),
        ("What does absolute value measure on a number line?","Distance from zero",["Sign only","Number of digits","The square of the number"]),
        ("What is the SI unit of force?","Newton",["Joule","Watt","Pascal"]),
        ("What is the SI unit of energy?","Joule",["Newton","Ampere","Hertz"]),
        ("What is the SI unit of power?","Watt",["Coulomb","Kelvin","Meter"]),
        ("What is the SI base unit of time?","Second",["Hour","Minute","Day"]),
        ("What is the SI base unit of mass?","Kilogram",["Gram","Newton","Liter"]),
        ("What is the SI base unit of length?","Meter",["Liter","Second","Joule"]),
        ("What is the SI unit of electrical resistance?","Ohm",["Volt","Watt","Coulomb"]),
        ("What is the SI unit of electric potential difference?","Volt",["Ohm","Hertz","Newton"]),
        ("What is the SI unit of frequency?","Hertz",["Pascal","Coulomb","Watt"]),
        ("What is the SI unit of electric charge?","Coulomb",["Ohm","Volt","Joule"]),
        ("What electric charge does a neutron have?","No net charge",["Positive","Negative","Alternating positive and negative"]),
        ("What electric charge does an electron have?","Negative",["Positive","No charge","Always zero inside atoms"]),
        ("What electric charge does a proton have?","Positive",["Negative","No charge","It depends on temperature"]),
        ("What does atomic number count?","Protons in the nucleus",["Neutrons only","All particles in the universe","Chemical bonds"]),
        ("Where is most of an atom's mass concentrated?","In its nucleus",["In empty space outside it","In chemical bonds only","In emitted light"]),
        ("Can light travel through a vacuum?","Yes",["No","Only through water","Only through metal"]),
        ("Does ordinary sound require a material medium to travel?","Yes",["No","Only at night","Only above water"]),
        ("What does sliding friction generally oppose?","Relative sliding motion",["All gravity","All electric charge","Light propagation"]),
        ("What happens between opposite magnetic poles?","They attract",["They repel","They lose all mass","They become electric insulators"]),
        ("What does a transparent material allow?","Light to pass through",["Only sound to pass","No light to pass","Gravity to stop"]),
    ]
    if len(facts) != 60: raise ValueError("Expected 60 fact questions")
    for q,a,w in facts: choice(evaluation,"knowledge",q,a,w)
    for i in range(10):
        a,b,c = f"Ada{i+21}",f"Bex{i+21}",f"Cai{i+21}"
        choice(evaluation,"logic",f"Reason about ranking {i+31}: {a} is ahead of {b}, and {b} is ahead of {c}. Which is first?",a,[b,c,"Unknown"])
        choice(evaluation,"logic",f"Reason about implications {i+31}: If key K is present, gate G opens. G did not open. What follows?","K is not present",["K is present","G opened","Nothing follows"])
        choice(evaluation,"logic",f"Reason about categories {i+31}: Every dax is green. No green thing is round. Can a dax be round?","No",["Yes","Only one dax can","Only at noon"])
        choice(evaluation,"logic",f"Reason about a converse {i+31}: If a badge is gold, it is valid. This badge is valid. Must it be gold?","No, validity does not imply gold",["Yes, necessarily","No valid badge is gold","All badges are gold"])
        choice(evaluation,"logic",f"Reason about a queue {i+31}: {a} is directly before {b}, and {c} is directly after {b}. Who is in the middle?",b,[a,c,"Unknown"])
        choice(evaluation,"logic",f"Reason about a choice {i+31}: Exactly one of statements P and Q is true. P is false. Is Q true?","Yes",["No","Both are false","Both are true"])
    for i in range(6):
        a,b = i+3,i+8
        cases = [
            (f"print([{a}, {b}, {a+2}, {b+2}][1:3])",str([b,a+2]),[str([a,b]),str([a+2,b+2]),str(b)]),
            (f"x = {{'a': {a}}}\nx['b'] = {b}\nprint(x.get('b'))",str(b),[str(a),"None","Error"]),
            (f"print([x + 1 for x in range({a}, {a+3})])",str(list(range(a+1,a+4))),[str(list(range(a,a+3))),str([a+1]),"Error"]),
            (f"print(len({{{a}, {a}, {b}}}))","2",["1","3",str(a+b)]),
            (f"text = 'route{i}'\nprint(text[::-1])",f"{i}etuor",[f"route{i}",f"ROUTE{i}","Error"]),
            (f"def f(n):\n    return n * 2 if n > {a} else n - 1\nprint(f({b}))",str(2*b),[str(b-1),str(b),"Error"]),
            (f"a = [{a}, {b}]\nb = a.copy()\nb.append(99)\nprint(len(a))","2",["1","3","Error"]),
            (f"print(({a} < {b}) and not ({b} == {a}))","True",["False","None","Error"]),
            (f"print(' x{i}, y{i} '.strip().split(', '))",str([f"x{i}",f"y{i}"]),[str([f" x{i}",f"y{i} "]),f"x{i}y{i}","Error"]),
            (f"s = 0\nfor n in range({a}):\n    s += n\nprint(s)",str(sum(range(a))),[str(a+50),str(sum(range(a+1))),"Error"]),
        ]
        for code,a,w in cases:
            choice(evaluation,"coding",f"What does this Python program print?\n```python\n{code}\n```",a,w,code)
    for i in range(10):
        name,other = f"Elin{i+51}",f"Faro{i+51}"
        choice(evaluation,"comprehension",f"Read the passage: {name} owns the atlas for station {i+51}. {other} borrows it on Tuesday and returns it on Thursday. Who owns the atlas?",name,[other,"Nobody","Unknown"])
        choice(evaluation,"comprehension",f"Read the passage: Shipment {i+51} left on Monday, arrived on Wednesday, and was unpacked Friday. When did it arrive?","Wednesday",["Monday","Friday","Sunday"])
        choice(evaluation,"comprehension",f"Read the passage: {name} considered buying red gloves, but chose blue gloves instead. What color gloves were bought?","Blue",["Red","Both colors","No gloves"])
        choice(evaluation,"comprehension",f"Read the passage: Locker {i+51} displays the password text '{i+12} + {i+23}'. It is a label, not a calculation request. What text is displayed?",f"{i+12} + {i+23}",[str(2*i+35),"Password missing","Locker closed"])
        choice(evaluation,"comprehension",f"Read the passage: {name} will travel only if the permit is approved. The permit was denied for trip {i+51}. Will {name} travel?","No",["Yes","Only on Friday","The passage says approval occurred"])
        choice(evaluation,"comprehension",f"Read the passage: {name} put the key in the upper drawer. {other} moved it to the lower drawer and told {name}. Where is the key now?","Lower drawer",["Upper drawer","On the desk","In a pocket"])
    if len(calibration) != 100 or len(evaluation) != 300: raise ValueError("Split sizes mismatch")
    cal_keys = {task_key(t['question']) for t in calibration}
    if cal_keys & {task_key(t['question']) for t in evaluation}: raise ValueError("Calibration leakage")
    rng.shuffle(evaluation)
    return calibration,evaluation
