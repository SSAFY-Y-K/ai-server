"""Unit tests for server-side validation helpers in main.py."""

import asyncio

import pytest
from main import (
    _compute_actual_primary_size,
    _validate_size_constraints,
    _validate_test_cases,
    AlgorithmGenerationPlan,
)


# ── _compute_actual_primary_size ──────────────────────────────────────────────

class TestComputeActualPrimarySize:

    # ── GRAPH ─────────────────────────────────────────────────────────────────

    def test_graph_header_n_e(self):
        inp = "5 8\n1 2\n2 3\n3 4\n4 5\n5 1\n1 3\n2 4\n3 5\n"
        assert _compute_actual_primary_size(inp, "GRAPH") == 8

    def test_graph_e_larger(self):
        inp = "3 10\n" + "1 2\n" * 10
        assert _compute_actual_primary_size(inp, "GRAPH") == 10

    def test_graph_n_larger(self):
        inp = "100000 2\n1 2\n2 3\n"
        assert _compute_actual_primary_size(inp, "GRAPH") == 100000

    # ── GRID ──────────────────────────────────────────────────────────────────

    def test_grid_r_times_c(self):
        inp = "3 4\n1 2 3 4\n5 6 7 8\n9 10 11 12\n"
        assert _compute_actual_primary_size(inp, "GRID") == 12

    # ── ARRAY: Format A (N Q on same header line) ─────────────────────────────

    def test_array_format_a_q_dominant(self):
        inp = "5 100000\n1 2 3 4 5\n100000\n" + "1\n" * 100000
        assert _compute_actual_primary_size(inp, "ARRAY", max_q=100000) == 100000

    def test_array_format_a_n_dominant(self):
        inp = "200000 3\n" + " ".join(str(i) for i in range(200000)) + "\n3\n1\n2\n3\n"
        assert _compute_actual_primary_size(inp, "ARRAY", max_q=3) == 200000

    # ── ARRAY: Format B (array on one line, Q on line 2) ─────────────────────

    def test_array_format_b_q_found(self):
        inp = "5\n1 2 3 4 5\n3\n7\n12\n3\n"
        assert _compute_actual_primary_size(inp, "ARRAY", max_q=10) == 5  # max(5,3)=5

    def test_array_format_b_q_dominant(self):
        inp = "5\n1 2 3 4 5\n50000\n" + "1\n" * 50000
        assert _compute_actual_primary_size(inp, "ARRAY", max_q=100000) == 50000

    # ── ARRAY: Format C (array one element per line, Q on line N+1) ──────────

    def test_array_format_c_q_found(self):
        inp = "4\n10\n20\n30\n40\n3\n1\n2\n3\n"
        # N=4, array=lines[1..4], Q=lines[5]=3, max(4,3)=4
        assert _compute_actual_primary_size(inp, "ARRAY", max_q=10) == 4

    def test_array_format_c_q_dominant(self):
        inp = "3\n10\n20\n30\n9000\n" + "1\n" * 9000
        # N=3, array=3 lines, Q=lines[4]=9000
        assert _compute_actual_primary_size(inp, "ARRAY", max_q=10000) == 9000

    # ── Ambiguous / fallback to N ──────────────────────────────────────────────

    def test_no_max_q_returns_n(self):
        """Without max_q hint, always return N."""
        inp = "5\n1 2 3 4 5\n3\n1\n2\n3\n"
        assert _compute_actual_primary_size(inp, "ARRAY", max_q=None) == 5

    def test_ambiguous_falls_back_to_n(self):
        """Multi-token second line that doesn't cover N elements → ambiguous → N."""
        inp = "100\n1 2 3\nsome text here\n"
        result = _compute_actual_primary_size(inp, "ARRAY", max_q=50)
        assert result == 100

    def test_empty_input_returns_none(self):
        assert _compute_actual_primary_size("", "ARRAY") is None

    def test_whitespace_only_returns_none(self):
        assert _compute_actual_primary_size("   \n  \n", "ARRAY") is None

    # ── ANSWER_SEARCH (same heuristic as ARRAY) ────────────────────────────────

    def test_answer_search_format_b(self):
        inp = "10\n1 2 3 4 5 6 7 8 9 10\n5\n1\n2\n3\n4\n5\n"
        assert _compute_actual_primary_size(inp, "ANSWER_SEARCH", max_q=20) == 10

    # ── INTERVAL / STRING: no Q heuristic ──────────────────────────────────────

    def test_interval_returns_n_only(self):
        inp = "5\n1 3\n2 5\n4 7\n6 9\n8 10\n"
        assert _compute_actual_primary_size(inp, "INTERVAL", max_q=999) == 5

    def test_string_returns_n_only(self):
        inp = "10\nabcdefghij\n"
        assert _compute_actual_primary_size(inp, "STRING", max_q=999) == 10


# ── input_data + input_generator mutual exclusion ────────────────────────────

class TestValidateTestCasesExclusivity:
    """Verify that having both input_data and input_generator is rejected."""

    def _make_plan(self):
        from main import AlgorithmGenerationPlan
        return AlgorithmGenerationPlan(
            difficulty="MEDIUM",
            category="구현",
            input_model="ARRAY",
            intended_algorithm="brute force",
            intended_time_complexity="O(N)",
            intended_memory_complexity="O(N)",
            opt_tc="LINEAR",
            bf_tc="N2",
            max_constraints="1 <= N <= 1000",
            max_n=1000,
            max_q=None,
            max_v=None,
            max_e=None,
            max_r=None,
            max_c=None,
            peak_state_items=1000,
            chosen_time_limit_ms=2000,
            chosen_memory_limit_mb=256,
            why_bruteforce_fails="N^2 too slow",
        )

    def _base_case(self, **overrides):
        base = {
            "input_data": "3\n1 2 3\n",
            "input_generator": "",
            "expected_output": "6",
            "primary_size": 3,
            "is_boundary": False,
            "is_anti_naive": False,
            "design_note": "test",
        }
        base.update(overrides)
        return base

    def test_both_present_is_rejected(self):
        from main import _validate_test_cases
        plan = self._make_plan()
        cases = [
            self._base_case(input_data="3\n1 2 3\n", input_generator="print(3)\nprint('1 2 3')", is_boundary=True),
            self._base_case(input_data="5\n1 2 3 4 5\n", is_anti_naive=True),
            self._base_case(input_data="2\n1 2\n", is_anti_naive=True),
            self._base_case(input_data="1\n1\n", is_boundary=True),
        ]
        issues = _validate_test_cases(plan, cases)
        assert any("both input_data and input_generator" in i for i in issues)

    def test_only_input_data_is_ok(self):
        from main import _validate_test_cases
        plan = self._make_plan()
        cases = [
            self._base_case(is_boundary=True),
            self._base_case(input_data="5\n1 2 3 4 5\n", primary_size=5, is_anti_naive=True),
            self._base_case(input_data="2\n1 2\n", primary_size=2, is_anti_naive=True),
            self._base_case(input_data="1\n1\n", primary_size=1, is_boundary=True),
        ]
        issues = _validate_test_cases(plan, cases)
        assert not any("both input_data and input_generator" in i for i in issues)


# ── N=0 edge cases ────────────────────────────────────────────────────────────

class TestNZeroEdgeCases:
    """N=0 returns 0 (not None) — parse succeeded, but policy rejects it."""

    def test_n0_q_on_next_line(self):
        inp = "0\n5000\n1\n2\n3\n4\n5000\n"
        assert _compute_actual_primary_size(inp, "ARRAY", max_q=10000) == 5000

    def test_n0_no_q_returns_zero(self):
        inp = "0\n"
        assert _compute_actual_primary_size(inp, "ARRAY", max_q=100) == 0

    def test_n0_no_max_q_returns_zero(self):
        """Without max_q, N=0 → return n=0 (no heuristic applied)."""
        inp = "0\n5000\n1\n2\n"
        assert _compute_actual_primary_size(inp, "ARRAY", max_q=None) == 0

    def test_n0_format_a_header(self):
        inp = "0 5000\n1\n2\n3\n"
        assert _compute_actual_primary_size(inp, "ARRAY", max_q=10000) == 5000

    def test_n0_parse_fails_returns_none(self):
        """Completely unparseable input → None."""
        assert _compute_actual_primary_size("not a number\n", "ARRAY") is None

    def test_empty_returns_none(self):
        assert _compute_actual_primary_size("", "ARRAY") is None

    def test_whitespace_returns_none(self):
        assert _compute_actual_primary_size("   \n  \n", "ARRAY") is None


# ── _validate_size_constraints ────────────────────────────────────────────────

def _make_plan_for_size(
    input_model: str = "ARRAY",
    difficulty: str = "MEDIUM",
    max_n: int = 200000,
    max_q: int | None = None,
    bf_tc: str = "N2",
    category: str = "구현",
) -> AlgorithmGenerationPlan:
    return AlgorithmGenerationPlan(
        difficulty=difficulty,
        category=category,
        input_model=input_model,
        intended_algorithm="test",
        intended_time_complexity="O(N)",
        intended_memory_complexity="O(N)",
        opt_tc="LINEAR",
        bf_tc=bf_tc,
        max_constraints=f"1 <= N <= {max_n}",
        max_n=max_n,
        max_q=max_q,
        max_v=None,
        max_e=None,
        max_r=None,
        max_c=None,
        peak_state_items=max_n,
        chosen_time_limit_ms=2000,
        chosen_memory_limit_mb=256,
        why_bruteforce_fails="too slow",
    )


def _tc(primary_size: int, is_anti_naive: bool = False) -> dict:
    return {
        "input_data": "placeholder",
        "input_generator": "",
        "expected_output": "1",
        "primary_size": primary_size,
        "is_boundary": False,
        "is_anti_naive": is_anti_naive,
        "design_note": "",
    }


class TestValidateSizeConstraints:

    def test_passes_when_near_max_met(self):
        """At least one case at 60% of 200000 → no issues."""
        plan = _make_plan_for_size()
        cases = [_tc(120000), _tc(50), _tc(10), _tc(5)]
        issues = _validate_size_constraints(plan, cases, 200000, 120000, 0)
        assert not any("close enough" in i for i in issues)

    def test_fails_when_no_near_max(self):
        """All cases below threshold → near-max issue."""
        plan = _make_plan_for_size()
        cases = [_tc(100000), _tc(50), _tc(10), _tc(5)]
        issues = _validate_size_constraints(plan, cases, 200000, 120000, 0)
        assert any("close enough" in i for i in issues)

    def test_fails_when_anti_naive_too_small(self):
        """Anti-naive case exists but primary_size below threshold."""
        plan = _make_plan_for_size(bf_tc="N_LOG_N", category="dp")
        cases = [_tc(120000), _tc(50, is_anti_naive=True), _tc(10), _tc(5)]
        threshold = 120000
        anti_threshold = 60000  # e.g. 30% of 200000
        issues = _validate_size_constraints(plan, cases, 200000, threshold, anti_threshold)
        assert any("anti-naive" in i for i in issues)

    def test_passes_when_anti_naive_large_enough(self):
        plan = _make_plan_for_size(bf_tc="N_LOG_N", category="dp")
        cases = [_tc(120000), _tc(80000, is_anti_naive=True), _tc(10), _tc(5)]
        issues = _validate_size_constraints(plan, cases, 200000, 120000, 60000)
        assert not any("anti-naive" in i for i in issues)

    def test_fails_when_case_exceeds_target(self):
        """A case claims primary_size larger than plan's max."""
        plan = _make_plan_for_size()
        cases = [_tc(300000), _tc(50), _tc(10), _tc(5)]
        issues = _validate_size_constraints(plan, cases, 200000, 120000, 0)
        assert any("exceeds the planned dominant size" in i for i in issues)

    def test_target_zero_skips_all_size_checks(self):
        """target_size=0 means no constraints to check."""
        plan = _make_plan_for_size()
        cases = [_tc(0), _tc(0), _tc(0), _tc(0)]
        issues = _validate_size_constraints(plan, cases, 0, 0, 0)
        assert issues == []

    def test_generator_materialization_revalidation(self):
        """Simulate what happens after generator overrides primary_size.

        LLM claimed primary_size=100000 (would pass), but actual generator
        output is tiny → server recalculates primary_size=10 → re-validation
        should now fail the near-max check.
        """
        plan = _make_plan_for_size()
        cases = [_tc(10), _tc(8), _tc(5), _tc(3, is_anti_naive=True)]
        target, near_max, anti = 200000, 120000, 40000
        issues = _validate_size_constraints(plan, cases, target, near_max, anti)
        assert any("close enough" in i for i in issues), \
            "Re-validation must catch small actual inputs even if LLM reported large primary_size"

    def test_generator_parse_failure_propagates_as_none(self):
        """Regression: generator produces unparseable output → primary_size = None.

        Before the fix, actual_size=None was skipped (is not None guard), so
        LLM's original large primary_size survived → near-max check passed incorrectly.
        After the fix, primary_size is unconditionally overridden to None → _safe_size
        treats it as 0 → near-max check fails as expected.
        """
        plan = _make_plan_for_size()
        # Simulate: LLM reported ps=200000; generator ran but _compute_actual_primary_size
        # returned None (unparseable output) → primary_size overridden to None
        cases = [_tc(None), _tc(None), _tc(None, is_anti_naive=True), _tc(None)]
        target, near_max, anti = 200000, 120000, 40000
        issues = _validate_size_constraints(plan, cases, target, near_max, anti)
        assert any("close enough" in i for i in issues), \
            "None primary_size must not count toward near-max (regression: stale LLM value must not survive)"

    def test_none_and_small_mix_still_fails(self):
        """Mix of None and small primary_sizes — neither reaches near_max_threshold."""
        plan = _make_plan_for_size()
        cases = [_tc(None), _tc(50), _tc(30, is_anti_naive=True), _tc(None)]
        issues = _validate_size_constraints(plan, cases, 200000, 120000, 0)
        assert any("close enough" in i for i in issues)

    def test_none_does_not_falsely_exceed_target(self):
        """None primary_size must not trigger 'exceeds planned dominant size'."""
        plan = _make_plan_for_size()
        cases = [_tc(None), _tc(120000), _tc(80000, is_anti_naive=True), _tc(50)]
        issues = _validate_size_constraints(plan, cases, 200000, 120000, 0)
        assert not any("exceeds" in i for i in issues), \
            "None should be treated as 0, not as a huge number"


# ── N=0 and None policy: explicit reject in _validate_test_cases ──────────────

class TestZeroAndNonePrimaryReject:
    """Policy: primary_size=0 and primary_size=None are both rejected."""

    def _make_plan(self, input_model="ARRAY", max_n=1000):
        return AlgorithmGenerationPlan(
            difficulty="MEDIUM", category="구현", input_model=input_model,
            intended_algorithm="test", intended_time_complexity="O(N)",
            intended_memory_complexity="O(N)", opt_tc="LINEAR", bf_tc="N2",
            max_constraints=f"1 <= N <= {max_n}", max_n=max_n, max_q=None,
            max_v=None, max_e=None, max_r=None, max_c=None,
            peak_state_items=max_n, chosen_time_limit_ms=2000,
            chosen_memory_limit_mb=256, why_bruteforce_fails="too slow",
        )

    def _tc_with_ps(self, ps, **kwargs):
        base = {
            "input_data": "3\n1 2 3\n",
            "input_generator": "",
            "expected_output": "6",
            "primary_size": ps,
            "is_boundary": False,
            "is_anti_naive": False,
            "design_note": "",
        }
        base.update(kwargs)
        return base

    def test_literal_zero_input_is_rejected(self):
        """Literal input '0\\n' computes primary_size=0 → validator rejects."""
        plan = self._make_plan()
        cases = [
            self._tc_with_ps(5000, input_data="0\n", is_boundary=True),  # N=0, LLM lied ps=5000
            self._tc_with_ps(3, input_data="3\n1 2 3\n", is_anti_naive=True),
            self._tc_with_ps(2, input_data="2\n1 2\n", is_anti_naive=True),
            self._tc_with_ps(1, input_data="1\n1\n", is_boundary=True),
        ]
        issues = _validate_test_cases(plan, cases)
        assert any("zero dominant size" in i for i in issues)

    def test_unparseable_input_is_rejected(self):
        """input_data that fails to parse → primary_size becomes None → validator rejects."""
        plan = self._make_plan()
        cases = [
            self._tc_with_ps(9999, input_data="not a number\n", is_boundary=True),
            self._tc_with_ps(3, input_data="3\n1 2 3\n", is_anti_naive=True),
            self._tc_with_ps(2, input_data="2\n1 2\n", is_anti_naive=True),
            self._tc_with_ps(1, input_data="1\n1\n", is_boundary=True),
        ]
        issues = _validate_test_cases(plan, cases)
        assert any("could not be inferred" in i for i in issues)

    def test_n0_with_q_is_accepted_size_wise(self):
        """'0 5000\\n...' → primary_size=5000 (Q dominant), passes size compute."""
        result = _compute_actual_primary_size("0 5000\n1\n2\n", "ARRAY", max_q=10000)
        assert result == 5000

    def test_ambiguous_input_returns_none(self):
        """Completely unparseable → None → caller should reject."""
        result = _compute_actual_primary_size("not a number at all\n", "ARRAY")
        assert result is None

    def test_generator_parse_failure_overrides_stale_primary_size(self, monkeypatch):
        """Generated input parse failure must replace stale LLM primary_size with None."""
        import main as main_module

        plan = self._make_plan()
        cases = [
            self._tc_with_ps(5000, input_data="", input_generator="print('bad')", is_boundary=True),
            self._tc_with_ps(700, input_data="3\n1 2 3\n", is_anti_naive=True),
            self._tc_with_ps(650, input_data="2\n1 2\n", is_anti_naive=True),
            self._tc_with_ps(900, input_data="1\n1\n", is_boundary=True),
        ]

        async def fake_run_generator(code, timeout=15.0):
            return "not a number\n", ""

        async def fake_run_solution(code, input_data, timeout=5.0):
            return "1", ""

        monkeypatch.setattr(main_module, "_run_generator", fake_run_generator)
        monkeypatch.setattr(main_module, "_run_solution", fake_run_solution)

        issues, _ = asyncio.run(
            main_module._materialize_reference_outputs("print(1)", cases, plan)
        )

        assert cases[0]["primary_size"] is None
        assert any("could not be inferred from generated input" in i for i in issues)

    def test_generator_failure_skips_reference_execution(self, monkeypatch):
        """If a generator already failed, do not run the reference solution at all."""
        import main as main_module

        plan = self._make_plan()
        cases = [
            self._tc_with_ps(500, input_data="", input_generator="print('boom')", is_boundary=True),
            self._tc_with_ps(700, input_data="3\n1 2 3\n", is_anti_naive=True),
            self._tc_with_ps(650, input_data="2\n1 2\n", is_anti_naive=True),
            self._tc_with_ps(900, input_data="1\n1\n", is_boundary=True),
        ]

        async def fake_run_generator(code, timeout=15.0):
            return None, "exit 1: IndexError: list index out of range"

        async def fake_run_solution(code, input_data, timeout=5.0):
            pytest.fail("reference solution should not run when generator materialization already failed")

        monkeypatch.setattr(main_module, "_run_generator", fake_run_generator)
        monkeypatch.setattr(main_module, "_run_solution", fake_run_solution)

        issues, _ = asyncio.run(
            main_module._materialize_reference_outputs("print(1)", cases, plan)
        )

        assert "generator_broken" in issues
        assert any("input generator failed on case 1" in i for i in issues)

    def test_reference_timeout_scales_with_plan(self, monkeypatch):
        """Reference execution timeout should scale with the plan time limit."""
        import main as main_module

        plan = self._make_plan()
        plan.chosen_time_limit_ms = 2500
        cases = [
            self._tc_with_ps(700, input_data="3\n1 2 3\n", is_boundary=True),
            self._tc_with_ps(650, input_data="2\n1 2\n", is_anti_naive=True),
            self._tc_with_ps(600, input_data="4\n1 2 3 4\n", is_anti_naive=True),
            self._tc_with_ps(900, input_data="1\n1\n", is_boundary=True),
        ]
        observed_timeouts: list[float] = []

        async def fake_run_solution(code, input_data, timeout=5.0):
            observed_timeouts.append(timeout)
            return "1", ""

        monkeypatch.setattr(main_module, "_run_solution", fake_run_solution)

        issues, _ = asyncio.run(
            main_module._materialize_reference_outputs("print(1)", cases, plan)
        )

        assert issues == []
        assert observed_timeouts == [7.5, 7.5, 7.5, 7.5]
