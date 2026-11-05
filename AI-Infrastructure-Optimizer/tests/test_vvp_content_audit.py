import unittest
from src.vvp_content_audit import interpret

class TestSupplementalContent(unittest.TestCase):
    def test_explicit_content_without_gold(self):
        q='What is printed?\nA) [1, 3]\nB) [2, 4]\nC) 6\nD) Error'
        self.assertEqual(interpret(q,'A.','choice')[0],'A')
        self.assertEqual(interpret(q,'The Python program outputs the list [1, 3]. Therefore, the correct option is A.','choice')[0],'A')
        self.assertEqual(interpret(q,'The Python program outputs option C: 6.','choice')[0],'C')
        self.assertEqual(interpret(q,'The Python program outputs the following:\n\nA) [1, 3]\n\nThis is because','choice')[0],'A')

    def test_conflicting_or_unfinished_answer_is_not_recovered(self):
        q='Task\nA) [1, 3]\nB) [2, 4]\nC) 6\nD) Error'
        self.assertIsNone(interpret(q,'The Python program outputs the list [1, 3]. Therefore, the correct option is B.','choice')[0])
        self.assertIsNone(interpret(q,'To determine this, first consider A','choice')[0])
        self.assertIsNone(interpret(q,'The Python program outputs option C: 9.','choice')[0])

if __name__=='__main__':unittest.main()
