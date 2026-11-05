"""Corrected additive entry point: union-column export, unchanged frozen model code."""
import sys
from .alsr_io import write_union

def run(setup_path):
    from . import alsr_experiment
    # Dependency injection affects exports in this process only, no source edits.
    alsr_experiment.write_records=write_union
    return alsr_experiment.run(setup_path)

if __name__=='__main__':run(sys.argv[1])
