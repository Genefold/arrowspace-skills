from __future__ import annotations

import unittest

import numpy as np

from arrowspace_skills import (
    build_index,
    explain_spectral_properties,
    item_lambdas,
    search_with_recall,
    sorted_lambdas,
    spectral_summary,
    suggest_params,
    tune_tau,
)


def _make_small_dataset() -> np.ndarray:
    rng = np.random.default_rng(3407)
    return rng.normal(size=(30, 4)).astype(np.float64)


class TestSuggestParams(unittest.TestCase):
    def test_large_dataset(self) -> None:
        p = suggest_params(10000, 768)
        self.assertEqual(p["eps"], 1.0)
        self.assertGreaterEqual(p["k"], 12)
        self.assertLessEqual(p["k"], 25)
        self.assertEqual(p["topk"], 6)
        self.assertEqual(p["p"], 2.0)

    def test_small_dataset(self) -> None:
        p = suggest_params(50, 128)
        self.assertEqual(p["eps"], 0.5)
        self.assertGreaterEqual(p["k"], 12)

    def test_low_dims(self) -> None:
        p = suggest_params(500, 64)
        self.assertEqual(p["eps"], 0.5)

    def test_high_dims(self) -> None:
        p = suggest_params(500, 1024)
        self.assertEqual(p["eps"], 2.0)


class TestBuildIndex(unittest.TestCase):
    def test_builds_index(self) -> None:
        items = _make_small_dataset()
        aspace, gl = build_index(items)
        self.assertEqual(aspace.nitems, 30)
        self.assertEqual(aspace.nfeatures, 4)
        self.assertGreater(gl.nnodes, 0)

    def test_rejects_1d_array(self) -> None:
        with self.assertRaises(ValueError):
            build_index(np.array([1.0, 2.0, 3.0]))

    def test_accepts_custom_params(self) -> None:
        items = _make_small_dataset()
        aspace, gl = build_index(items, {"eps": 2.0, "k": 4, "topk": 2, "p": 1.0, "sigma": 0.5})
        self.assertEqual(aspace.nitems, 30)

    def test_converts_dtype(self) -> None:
        items = np.random.default_rng(3407).normal(size=(10, 3)).astype(np.float32)
        # eps must suit this tiny random dataset (cosine distances ~sqrt(2))
        aspace, gl = build_index(items, {"eps": 2.0, "k": 4, "topk": 2, "p": 2.0, "sigma": None})
        self.assertEqual(aspace.nitems, 10)


class TestSearchWithRecall(unittest.TestCase):
    def setUp(self) -> None:
        self.items = _make_small_dataset()
        self.aspace, self.gl = build_index(self.items)
        self.query = self.items[0]

    def test_returns_hits(self) -> None:
        hits = search_with_recall(self.aspace, self.gl, self.query, tau=1.0)
        self.assertGreater(len(hits), 0)
        for idx, score in hits:
            self.assertIsInstance(idx, int)
            self.assertIsInstance(score, float)

    def test_max_results_respected(self) -> None:
        hits = search_with_recall(self.aspace, self.gl, self.query, tau=1.0, max_results=2)
        self.assertLessEqual(len(hits), 2)

    def test_max_results_default(self) -> None:
        hits = search_with_recall(self.aspace, self.gl, self.query, tau=1.0)
        self.assertGreater(len(hits), 0)


class TestTuneTau(unittest.TestCase):
    def setUp(self) -> None:
        self.items = _make_small_dataset()
        self.aspace, self.gl = build_index(self.items)
        rng = np.random.default_rng(3407)
        self.queries = rng.normal(size=(3, 4)).astype(np.float64)
        self.ground_truth = [[0], [1], [2]]

    def test_returns_float(self) -> None:
        tau = tune_tau(self.aspace, self.gl, self.queries, self.ground_truth)
        self.assertIsInstance(tau, float)

    def test_rejects_empty_ground_truth(self) -> None:
        with self.assertRaises(ValueError):
            tune_tau(self.aspace, self.gl, self.queries, [[], [], []])


class TestSpectralProperties(unittest.TestCase):
    def setUp(self) -> None:
        self.items = _make_small_dataset()
        self.aspace, self.gl = build_index(self.items)

    def test_explain_returns_dict(self) -> None:
        props = explain_spectral_properties(self.gl)
        for key in ("n_nodes", "eigval_min", "eigval_max", "fiedler_value", "spectral_gap", "condition_number_estimate"):
            self.assertIn(key, props)
        self.assertEqual(props["n_nodes"], self.gl.nnodes)

    def test_rejects_large_graph(self) -> None:
        rng = np.random.default_rng(3407)
        large_items = rng.normal(size=(3000, 4)).astype(np.float64)
        _, large_gl = build_index(large_items, {"eps": 1.0, "k": 6, "topk": 3, "p": 2.0, "sigma": 1.0})
        with self.assertRaises(ValueError):
            explain_spectral_properties(large_gl)

    def test_spectral_summary_string(self) -> None:
        summary = spectral_summary(self.gl)
        self.assertIsInstance(summary, str)
        self.assertIn("Nodes", summary)
        self.assertIn("Condition number estimate", summary)


class TestItemLambdas(unittest.TestCase):
    def setUp(self) -> None:
        self.items = _make_small_dataset()
        self.aspace, self.gl = build_index(self.items)

    def test_returns_array(self) -> None:
        scores = item_lambdas(self.aspace)
        self.assertIsInstance(scores, np.ndarray)
        self.assertEqual(scores.shape, (30,))

    def test_values_are_floats(self) -> None:
        scores = item_lambdas(self.aspace)
        for s in scores:
            self.assertIsInstance(s, (float, np.floating))


class TestSortedLambdas(unittest.TestCase):
    def setUp(self) -> None:
        self.items = _make_small_dataset()
        self.aspace, self.gl = build_index(self.items)

    def test_tuple_order_is_score_index(self) -> None:
        ranked = sorted_lambdas(self.aspace)
        self.assertGreater(len(ranked), 0)
        score, idx = ranked[0]
        self.assertIsInstance(score, float)
        self.assertIsInstance(idx, int)

    def test_sorted_ascending(self) -> None:
        ranked = sorted_lambdas(self.aspace)
        scores = [s for s, _ in ranked]
        for i in range(len(scores) - 1):
            self.assertLessEqual(scores[i], scores[i + 1])

    def test_30_items(self) -> None:
        ranked = sorted_lambdas(self.aspace)
        self.assertEqual(len(ranked), 30)


if __name__ == "__main__":
    unittest.main()
