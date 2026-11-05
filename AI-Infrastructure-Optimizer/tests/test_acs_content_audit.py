import unittest
from src.acs_content_audit import audit_fact

class TestACSContentAudit(unittest.TestCase):
    def test_synonym_coverage_is_distinct_from_added_facts(self):
        t={'score_kind':'constraints','required':['Dr. Leno','doctor','Studio 9'],'question':'Dr. Leno is a doctor. Dr. Leno works at Studio 9.'}
        a=audit_fact(t,'Dr. Leno, a dedicated physician, works at Studio 9.',{'content_correct':False,'incomplete_output':True})
        self.assertTrue(a['audited_fact_coverage']);self.assertTrue(a['synonym_recognized']);self.assertTrue(a['instruction_fidelity_known_failure'])
        b=audit_fact(t,'Dr.',{'content_correct':False,'incomplete_output':True});self.assertFalse(b['audited_fact_coverage']);self.assertTrue(b['audited_incomplete'])
    def test_doctorate_is_not_the_requested_role(self):
        t={'score_kind':'constraints','required':['Leno','doctor','Studio 9'],'question':'Leno is a doctor at Studio 9.'}
        a=audit_fact(t,'Leno has a doctorate at Studio 9.',{'content_correct':True,'incomplete_output':False});self.assertFalse(a['audited_fact_coverage'])
