import unittest

from src.answer_validator import extract_answer, validate_answer


class TestAnswerValidator(unittest.TestCase):
    def test_signed_integers_and_zero(self):
        for text, expected in [("12", 12), ("-12", -12), ("+12", 12), ("0", 0)]:
            with self.subTest(text=text):
                self.assertEqual(extract_answer(f"FINAL_ANSWER: {text}"), expected)

    def test_explanation_whitespace_and_case(self):
        self.assertEqual(extract_answer(
            "Calculation: 4 + 8 = 12\r\n\tfinal_answer : +12 \t\r\nDone."
        ), 12)

    def test_missing_or_invalid_input(self):
        for text in [None, 12, "", "The answer is 12", "12", "prefix FINAL_ANSWER: 12"]:
            with self.subTest(text=text):
                self.assertIsNone(extract_answer(text))

    def test_reject_malformed_integer(self):
        for value in ["12.0", "1/2", "1,000", "1e3", "12 extra", "１２", "", "--12"]:
            with self.subTest(value=value):
                self.assertIsNone(extract_answer(f"FINAL_ANSWER: {value}"))

    def test_reject_number_on_next_line(self):
        self.assertIsNone(extract_answer("FINAL_ANSWER:\n12"))

    def test_reject_multiple_declarations(self):
        for second in ["12", "13", "invalid"]:
            with self.subTest(second=second):
                self.assertIsNone(extract_answer(
                    f"FINAL_ANSWER: 12\nFINAL_ANSWER: {second}"
                ))

    def test_correct(self):
        self.assertEqual(validate_answer("FINAL_ANSWER: 12", 12), {
            "predicted": 12, "expected": 12, "correct": True, "status": "correct",
        })

    def test_incorrect(self):
        self.assertEqual(validate_answer("FINAL_ANSWER: 13", 12), {
            "predicted": 13, "expected": 12, "correct": False, "status": "incorrect",
        })

    def test_extraction_failed(self):
        self.assertEqual(validate_answer("The answer is 12", 12), {
            "predicted": None, "expected": 12,
            "correct": False, "status": "extraction_failed",
        })


if __name__ == "__main__":
    unittest.main()
