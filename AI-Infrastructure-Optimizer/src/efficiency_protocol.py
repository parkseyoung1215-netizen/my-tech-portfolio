"""Gold-independent content interpretation and independent raw-format compliance."""
import json
import re

def unwrap(text):
    text=text.strip()
    match=re.fullmatch(r'```(?:json|text)?[ \t]*\r?\n([\s\S]*?)\r?\n```',text,re.I)
    return (match.group(1).strip(),True) if match else (text,False)

def extract(question,response,kind):
    """No gold. Entire response must match one unambiguous accepted representation."""
    original=response.strip();text,fenced=unwrap(original)
    strict=bool(re.fullmatch(r'[A-D]',original) if kind=='choice' else re.fullmatch(r'[+-]?[0-9]+',original))
    def result(pred,mode):return {'prediction':pred,'format_compliant':strict,'parse_mode':mode}
    if kind=='choice':
        if re.fullmatch('[A-D]',text,re.I):return result(text.upper(),'fenced_bare' if fenced else 'bare')
        final=re.fullmatch(r'FINAL_ANSWER[ \t]*:[ \t]*([A-D])',text,re.I)
        if final:return result(final.group(1).upper(),'final_line')
        options=dict(re.findall(r'(?m)^([A-D])\) (.*)$',question))
        match=re.fullmatch(r'([A-D])\)[ \t]*(.*)',text,re.I)
        if match:
            letter,body=match.group(1).upper(),match.group(2)
            if letter in options and (not body or body==options[letter]):return result(letter,'option_notation')
        matches=[letter for letter,body in options.items() if body==text]
        if len(matches)==1:return result(matches[0],'exact_option_text')
    elif kind=='integer':
        final=re.fullmatch(r'FINAL_ANSWER[ \t]*:[ \t]*([+-]?[0-9]+)',text,re.I)
        token=final.group(1) if final else text
        if re.fullmatch(r'[+-]?[0-9]+',token):
            try:return result(str(int(token)),'final_line' if final else ('fenced_bare' if fenced else 'bare'))
            except ValueError:pass
    return {'prediction':None,'format_compliant':False,'parse_mode':'unresolved'}

def unique_object(pairs):
    obj={}
    for key,value in pairs:
        if key in obj:raise ValueError('Duplicate JSON key')
        obj[key]=value
    return obj

def extract_batch(response,questions,kinds):
    if not questions or len(questions)!=len(kinds):raise ValueError('Nonempty aligned batch required')
    text,fenced=unwrap(response);error=''
    try:
        obj=json.loads(text,object_pairs_hook=unique_object)
        if not isinstance(obj,dict) or set(obj)!={'answers'}:raise ValueError('Expected only answers object')
        answers=obj['answers']
        if not isinstance(answers,dict) or set(answers)!={f'Q{i}' for i in range(len(questions))}:raise ValueError('Missing/extra IDs')
    except (ValueError,TypeError) as exc:answers={};error=str(exc)
    items=[]
    for i,(q,kind) in enumerate(zip(questions,kinds)):
        value=answers.get(f'Q{i}')
        raw=str(value) if isinstance(value,(str,int)) and not isinstance(value,bool) else ''
        item=extract(q,raw,kind)
        item['format_compliant']=bool(not fenced and not error and item['format_compliant'])
        item['raw_value']=raw;items.append(item)
    return {'items':items,'envelope_fenced':fenced,'error':error,
            'format_compliant':all(item['format_compliant'] for item in items) and not error}
