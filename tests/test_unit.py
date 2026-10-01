import unittest

from evaluate import answer_scores, joint_metrics, prompt, support_scores


class EvaluationContractTest(unittest.TestCase):
    def test_prompt_contract(self):
        value = prompt("Who?", ["[A#0] Alice"], "supporting_facts")
        self.assertIn("Question: Who?", value)
        self.assertIn("supporting_facts", value)

    def test_answer_f1(self):
        self.assertEqual(answer_scores("The Alder River", "Alder River")[3], 1.0)

    def test_joint_f1(self):
        metrics = joint_metrics(answer_scores("yes", "yes"), support_scores({1}, {1}))
        self.assertEqual(metrics["joint_f1"], 1.0)


if __name__ == "__main__":
    unittest.main()
