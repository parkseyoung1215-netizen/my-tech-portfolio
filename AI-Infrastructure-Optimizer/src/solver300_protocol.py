"""Gold-independent answer parsing fixed before the 300-task experiment."""
import re


def content_extract(response,kind):
    declarations=[line for line in response.splitlines() if re.match(r'^[ \t]*FINAL_ANSWER\b',line,re.I)]
    token=response.strip()
    mode='bare_answer'
    if declarations:
        if len(declarations)!=1: return None,'unresolved'
        match=re.fullmatch(r'[ \t]*FINAL_ANSWER[ \t]*:[ \t]*([A-D]|[+-]?[0-9]+)[ \t]*',declarations[0],re.I)
        if not match: return None,'unresolved'
        token=match.group(1);mode='final_line'
    if kind=='choice' and re.fullmatch('[A-D]',token,re.I): return token.upper(),mode
    if kind=='integer' and re.fullmatch('[+-]?[0-9]+',token):
        try: return str(int(token)),mode
        except ValueError: pass
    return None,'unresolved'
