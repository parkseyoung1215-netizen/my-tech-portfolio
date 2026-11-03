
import unittest

from src.cost_analyzer import (
    calculate_cost,
    find_duplicate_requests,
    is_cache_safe,
    simulate_cache_savings,
)


class TestCostAnalyzer(unittest.TestCase):

    def test_cost_calculation(self):
        cost = calculate_cost("model_a", 1000, 500)
        self.assertAlmostEqual(cost, 0.003)

    def test_duplicate_detection(self):
        requests = [
            {"prompt": "What is AI?"},
            {"prompt": "what is ai?"},
            {"prompt": "Explain Python"},
        ]

        result = find_duplicate_requests(requests)

        self.assertEqual(result["what is ai?"], 2)

    def make_request(self, **overrides):
        request = {
            "tenant_id": "company_a",
            "user_id": "user_1",
            "model": "model_a",
            "model_version": "v1",
            "system_prompt_version": "v1",
            "prompt": "What is AI?",
            "input_tokens": 1000,
            "output_tokens": 500,
            "temperature": 0,
            "cacheable": True,
            "tools_enabled": False,
            "requires_fresh_data": False,
        }

        request.update(overrides)
        return request

    def test_safe_cache(self):
        request = self.make_request()
        self.assertTrue(is_cache_safe(request))

    def test_reject_unsafe_cache(self):
        request = self.make_request(temperature=0.7)
        self.assertFalse(is_cache_safe(request))

    def test_user_isolation(self):
        requests = [
            self.make_request(),
            self.make_request(user_id="user_2"),
        ]

        result = simulate_cache_savings(requests)

        self.assertEqual(result["cache_hits"], 0)

    def test_safe_cache_savings(self):
        requests = [
            self.make_request(),
            self.make_request(),
        ]

        result = simulate_cache_savings(requests)

        self.assertEqual(result["cache_hits"], 1)
        self.assertGreater(result["savings"], 0)


if __name__ == "__main__":
    unittest.main()
