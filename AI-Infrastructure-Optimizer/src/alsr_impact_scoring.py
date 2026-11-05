"""Offline only: retain the legacy score and reject JSON bool/number type confusion."""
from .alsr_dataset import grade as legacy_grade

def grade(task,response):
    result=legacy_grade(task,response)
    if task['score_kind']=='json' and result['correct']:
        prediction=result['prediction'];expected=task['expected_object']
        if any(type(prediction[k]) is not type(v) for k,v in expected.items()):
            result=dict(result,correct=False,error_type='json_value_type_mismatch')
    return result
