import unittest

from scode.orchestrator import run_parallel_tasks


class OrchestratorTests(unittest.TestCase):
    def test_parallel_results_preserve_prompt_order(self) -> None:
        prompts = ["a", "b", "c"]

        def runner(prompt: str) -> str:
            return prompt.upper()

        results = run_parallel_tasks(prompts, runner, max_workers=3)
        self.assertEqual([r.output for r in results], ["A", "B", "C"])

    def test_parallel_captures_task_failure(self) -> None:
        prompts = ["ok", "boom"]

        def runner(prompt: str) -> str:
            if prompt == "boom":
                raise RuntimeError("fail")
            return prompt

        results = run_parallel_tasks(prompts, runner, max_workers=2)
        self.assertEqual(results[0].output, "ok")
        self.assertIn("Agent failed:", results[1].output)

    def test_rejects_non_positive_workers(self) -> None:
        with self.assertRaises(ValueError):
            run_parallel_tasks(["x"], lambda x: x, max_workers=0)
