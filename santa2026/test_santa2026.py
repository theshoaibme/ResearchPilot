"""Unit tests for Santa 2026 validator and baseline solver."""

from __future__ import annotations

import unittest

from solver import solve_puzzle_baseline
from validator import Puzzle, validate_program


class TestSanta2026(unittest.TestCase):
    def test_valid_program(self) -> None:
        puzzle = Puzzle(id="t1", starting_values=(2, 3), targets=(5, 6, 11), value_limit=100, registers=48)
        prog = [
            ["+", 0, 1, 2],  # r2 = 5
            ["*", 0, 1, 3],  # r3 = 6
            ["+", 2, 3, 4],  # r4 = 11
        ]
        res = validate_program(puzzle, prog)
        self.assertTrue(res.valid, res.error)
        self.assertEqual(res.length, 3)

    def test_rejects_non_exact_division(self) -> None:
        puzzle = Puzzle(id="t2", starting_values=(5, 2), targets=(2,), value_limit=100, registers=48)
        res = validate_program(puzzle, [["/", 0, 1, 2]])
        self.assertFalse(res.valid)

    def test_rejects_value_limit_overflow(self) -> None:
        puzzle = Puzzle(id="t3", starting_values=(10, 10), targets=(100,), value_limit=50, registers=48)
        res = validate_program(puzzle, [["*", 0, 1, 2]])
        self.assertFalse(res.valid)

    def test_rejects_undefined_register(self) -> None:
        puzzle = Puzzle(id="t4", starting_values=(2,), targets=(4,), value_limit=100, registers=48)
        res = validate_program(puzzle, [["+", 0, 5, 1]])
        self.assertFalse(res.valid)

    def test_baseline_solver_produces_valid_programs(self) -> None:
        puzzle = Puzzle(
            id="t5",
            starting_values=(3, 7),
            targets=(1, 10, 21, 37, 100),
            value_limit=10**18,
            registers=48,
        )
        prog = solve_puzzle_baseline(puzzle)
        res = validate_program(puzzle, prog)
        self.assertTrue(res.valid, res.error)


if __name__ == "__main__":
    unittest.main()
