"""Request-only completion hypotheses, not a correctness oracle or calibrated confidence."""
import json,re
METHODS=('baseline','fixed8','fixed16','boundary','confidence','adaptive')

def request_kind(question):
    q=question.lower()
    if question.rstrip().endswith('Return only the option letter.'):return 'choice'
    if 'json' in q and ('object' in q or 'encode' in q):return 'json'
    if re.search(r'(?:one|single) (?:concise )?sentence',q):return 'sentence'
    return 'unknown'

def requested_keys(question):
    m=re.search(r'(?:with exactly|with keys|exactly keys|exactly these keys)\s+([a-zA-Z_, ]+?)(?:[.:;]|\n)',question,re.I)
    if not m:return []
    return [x for x in re.split(r'\s*(?:,|\band\b)\s*',m.group(1).strip()) if re.fullmatch('[A-Za-z_]+',x)]

def unique(pairs):
    obj={}
    for k,v in pairs:
        if k in obj:raise ValueError('Duplicate JSON key')
        obj[k]=v
    return obj

def boundary(question,text,adaptive=False):
    kind=request_kind(question);s=text.strip()
    if kind=='choice':return bool(re.fullmatch('[A-D]',s))
    if kind=='json':
        try:obj=json.loads(s,object_pairs_hook=unique)
        except (ValueError,TypeError):return False
        keys=requested_keys(question)
        return isinstance(obj,dict) and bool(obj) and (not keys or set(obj)==set(keys))
    if kind=='sentence':
        if not s or '\n' in s or not re.search(r'[.!?]["\']?$',s):return False
        if adaptive:
            if len(s.split())<6:return False
            if re.search(r'(?:\b(?:Dr|Mr|Mrs|Ms|Prof)|e\.g|i\.e)\.$',s,re.I):return False
        return True
    return False

def threshold(question,input_tokens,method):
    if method=='confidence':return .90
    base={'choice':.75,'json':.60,'sentence':.95,'unknown':1.0}[request_kind(question)]
    return min(.99,base+(.05 if input_tokens>128 else 0))

def limit(question,method):
    natural=16 if request_kind(question)=='choice' else 48
    return min(natural,8 if method=='fixed8' else 16 if method=='fixed16' else natural)
