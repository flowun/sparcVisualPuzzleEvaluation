import json
import unittest
import warnings

from evaluate_object_detection import extract_model_solution, is_same_poly, analyze_solution


class ExtractModelSolutionTests(unittest.TestCase):
    def parses_literal_list_with_code_fence(self):
        output = """```python\n[['a', 'b'], ['c', 'd']]\n```"""
        self.assertEqual(extract_model_solution(output), [["a", "b"], ["c", "d"]])

    def extracts_after_solution_marker(self):
        output = "prefix #### [[1, 2], [3, 4]]"
        self.assertEqual(extract_model_solution(output), [["1", "2"], ["3", "4"]])

    def returns_empty_list_when_no_brackets(self):
        output = "no array here"
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.assertEqual(extract_model_solution(output), [[]])
            self.assertTrue(any("Could not parse model solution" in str(w.message) for w in caught))

    def handles_extra_text_and_whitespace(self):
        output = "text before [ [ 'x' , 'y' ] , [ 'z' , 'w' ] ] trailing"
        self.assertEqual(extract_model_solution(output), [["x", "y"], ["z", "w"]])

    def parses_escaped_newlines_and_quotes(self):
        output = "####\\n[\\n  [\\'-G\\', \\'+\\'],\\n  [\\'S\\', \\'+\\']\\n]"
        self.assertEqual(extract_model_solution(output), [["-G", "+"], ["S", "+"]])

    def extract_large_solution(self):
        output = "#### [['S', '+', '+', '+', '+'], ['+', 'o-R', '+', 'N', '+'], ['+', '+', '.', '.', '+'], ['+', 'N', '+', 'N', '+'], ['+', '+', '+', '+', '+'], ['+', 'N', '+', 'N', '+'], ['+', '+', '+', '+', '+'], ['+', 'N', '+', 'N', '+'], ['+', '+', '+', '+', '+'], ['+', 'o-K', '+', 'o-P', '+'], ['+', '+', '+', '+', 'E']] "
        self.assertEqual(extract_model_solution(output), [['S', '+', '+', '+', '+'], ['+', 'o-R', '+', 'N', '+'], ['+', '+', '.', '.', '+'], ['+', 'N', '+', 'N', '+'], ['+', '+', '+', '+', '+'], ['+', 'N', '+', 'N', '+'], ['+', '+', '+', '+', '+'], ['+', 'N', '+', 'N', '+'], ['+', '+', '+', '+', '+'], ['+', 'o-K', '+', 'o-P', '+'], ['+', '+', '+', '+', 'E']])

    def warns_and_returns_empty_for_malformed_output(self):
        output = "#### not an array"
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.assertEqual(extract_model_solution(output), [[]])
            self.assertTrue(any("Could not parse model solution" in str(w.message) for w in caught))


class IsSamePolyTests(unittest.TestCase):
    def raises_value_error_for_invalid_prefix(self):
        poly_definitions = {"16": [[0,1,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]}
        with self.assertRaises(ValueError):
            is_same_poly("X-B-1", "P-B-16", poly_definitions)

    def returns_false_for_type_mismatch(self):
        poly_definitions = {"16": [[0,1,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]}
        self.assertFalse(is_same_poly("P-B-1", "Y-B-16", poly_definitions))

    def returns_false_for_color_mismatch(self):
        poly_definitions = {"16": [[0,1,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]}
        self.assertFalse(is_same_poly("P-R-1", "P-B-16", poly_definitions))

    def returns_false_for_oversized_shape_encoding(self):
        poly_definitions = {"16": [[0,1,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]}
        provided_cell = "P-B-" + "-".join(["0"] * 26)
        self.assertFalse(is_same_poly(provided_cell, "P-B-16", poly_definitions))

    def returns_false_for_invalid_shape_character(self):
        poly_definitions = {"16": [[0,1,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]}
        self.assertFalse(is_same_poly("P-B-X", "P-B-16", poly_definitions))

    def returns_true_for_matching_polys(self):
        poly_definitions = {"16": [[0,1,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]}
        self.assertTrue(is_same_poly("P-B-1", "P-B-16", poly_definitions))

    def returns_true_for_matching_polys2(self):
        poly_definitions = {"784": [[0,1,1,0],[0,0,1,0],[0,0,0,0],[0,0,0,0]]}
        self.assertTrue(is_same_poly("P-B-110-010-000", "P-B-784", poly_definitions))


class AnalyzeSolutionTests(unittest.TestCase):
    def computes_fraction_and_type_stats_with_mixed_cells(self):
        data = {
            "puzzle_array": [
                ["P-B-1", "G", "Y-R-2"],
                ["G", "P-B-1", "B"],
            ],
            "polyshapes": json.dumps({
                "1": [[1]],
                "2": [[0, 1], [0, 0]],
            }),
        }
        model_solution = [
            ["P-B-10-00", "G", "Y-R-01-00"],
            ["G", "P-B-0", "C"],
        ]

        is_fully_valid, valid_fraction, per_type_stats = analyze_solution(model_solution, data)

        self.assertEqual(is_fully_valid, 0)
        self.assertAlmostEqual(valid_fraction, 4 / 6)
        self.assertEqual(per_type_stats["P"], {"total": 2, "correct": 1, "fraction": 0.5})
        self.assertEqual(per_type_stats["G"], {"total": 2, "correct": 2, "fraction": 1.0})
        self.assertEqual(per_type_stats["Y"], {"total": 1, "correct": 1, "fraction": 1.0})
        self.assertEqual(per_type_stats["T"], {"total": 1, "correct": 0, "fraction": 0.0})

    def counts_missing_model_rows_as_incorrect_by_type(self):
        data = {
            "puzzle_array": [
                ["A", "B"],
                ["P-B-1", "A"],
            ],
            "polyshapes": json.dumps({"1": [[1]]}),
        }
        model_solution = [["A", "B"]]

        is_fully_valid, valid_fraction, per_type_stats = analyze_solution(model_solution, data)

        self.assertEqual(is_fully_valid, 0)
        self.assertAlmostEqual(valid_fraction, 0.5)
        self.assertEqual(per_type_stats["T"], {"total": 3, "correct": 2, "fraction": 2 / 3})
        self.assertEqual(per_type_stats["P"], {"total": 1, "correct": 0, "fraction": 0.0})

    def counts_missing_model_columns_as_incorrect_by_type(self):
        data = {
            "puzzle_array": [
                ["A", "B"],
                ["P-B-1", "A"],
            ],
            "polyshapes": json.dumps({"1": [[1]]}),
        }
        model_solution = [["A"], ["P-B-1"]]

        is_fully_valid, valid_fraction, per_type_stats = analyze_solution(model_solution, data)

        self.assertEqual(is_fully_valid, 0)
        self.assertAlmostEqual(valid_fraction, 0.5)
        self.assertEqual(per_type_stats["T"], {"total": 3, "correct": 1, "fraction": 1 / 3})
        self.assertEqual(per_type_stats["P"], {"total": 1, "correct": 1, "fraction": 1.0})

def load_tests(loader, tests, pattern):
    suite = unittest.TestSuite()
    for name in [
        "parses_literal_list_with_code_fence",
        "extracts_after_solution_marker",
        "returns_empty_list_when_no_brackets",
        "handles_extra_text_and_whitespace",
        "parses_escaped_newlines_and_quotes",
        "extract_large_solution",
        "warns_and_returns_empty_for_malformed_output",
    ]:
        suite.addTest(ExtractModelSolutionTests(name))
    for name in [
        "raises_value_error_for_invalid_prefix",
        "returns_false_for_type_mismatch",
        "returns_false_for_color_mismatch",
        "returns_false_for_oversized_shape_encoding",
        "returns_false_for_invalid_shape_character",
        "returns_true_for_matching_polys",
        "returns_true_for_matching_polys2",
    ]:
        suite.addTest(IsSamePolyTests(name))
    for name in [
        "computes_fraction_and_type_stats_with_mixed_cells",
        "counts_missing_model_rows_as_incorrect_by_type",
        "counts_missing_model_columns_as_incorrect_by_type",
    ]:
        suite.addTest(AnalyzeSolutionTests(name))
    return suite
