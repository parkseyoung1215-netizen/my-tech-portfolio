"""Fixed experiment factors and gold-independent requested-output completion."""
import re
VARIANTS={}
for backend in ('legacy','lean'):
    for method in ('full','stable1','sketch'):VARIANTS[backend+'_'+method]={'backend':backend,'method':method,'format_stop':False}
for method in ('full','stable1','sketch'):VARIANTS['lean_option_'+method]={'backend':'lean','method':method,'format_stop':True}

def option_complete(question,text):
    return question.rstrip().endswith('Return only the option letter.') and re.fullmatch('[A-D]',text.strip()) is not None
