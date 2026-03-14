from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


SUPPORTED_OPERATORS = {
    "add",
    "subtract",
    "multiply",
    "divide",
    "exp",
    "greater",
    "table_sum",
    "table_average",
    "table_min",
    "table_max",
}


@dataclass(frozen=True)
class ExecutionResult:
    """Container for execution outcomes."""

    executable: bool
    value: Any
    error: str


def looks_like_program(text: str) -> bool:
    """Return whether a text answer resembles a supported program expression."""
    stripped = text.strip()
    if not stripped:
        return False
    return bool(re.match(rf"^(?:{'|'.join(sorted(SUPPORTED_OPERATORS))})\(", stripped))


def split_top_level(text: str, delimiter: str = ",") -> list[str]:
    """Split text by a delimiter while respecting nested parentheses."""
    parts: list[str] = []
    current: list[str] = []
    depth = 0

    for char in text:
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1

        if char == delimiter and depth == 0:
            parts.append("".join(current).strip())
            current = []
            continue
        current.append(char)

    if current:
        parts.append("".join(current).strip())
    return [part for part in parts if part]


def _parse_literal(token: str, results: list[Any]) -> Any:
    """Parse a literal token into a numeric or referenced value."""
    token = token.strip()
    if not token:
        raise ValueError("empty_token")

    if token.startswith("#"):
        index = int(token[1:])
        if index >= len(results):
            raise ValueError(f"unknown_intermediate_reference:{token}")
        return results[index]

    if token.startswith("const_"):
        return float(token[len("const_") :].replace("_", ""))

    cleaned = token.replace("$", "").replace(",", "")
    if cleaned.endswith("%"):
        return float(cleaned[:-1]) / 100.0
    return float(cleaned)


def _apply_operator(name: str, args: list[Any]) -> Any:
    """Apply one supported program operator."""
    if name == "add":
        return sum(float(arg) for arg in args)
    if name == "subtract":
        if len(args) != 2:
            raise ValueError("subtract_requires_2_args")
        return float(args[0]) - float(args[1])
    if name == "multiply":
        value = 1.0
        for arg in args:
            value *= float(arg)
        return value
    if name == "divide":
        if len(args) != 2:
            raise ValueError("divide_requires_2_args")
        denominator = float(args[1])
        if denominator == 0:
            raise ZeroDivisionError("divide_by_zero")
        return float(args[0]) / denominator
    if name == "exp":
        if len(args) != 2:
            raise ValueError("exp_requires_2_args")
        return float(args[0]) ** float(args[1])
    if name == "greater":
        if len(args) != 2:
            raise ValueError("greater_requires_2_args")
        return "yes" if float(args[0]) > float(args[1]) else "no"
    if name == "table_sum":
        return sum(float(arg) for arg in args)
    if name == "table_average":
        if not args:
            raise ValueError("table_average_requires_args")
        return sum(float(arg) for arg in args) / len(args)
    if name == "table_min":
        if not args:
            raise ValueError("table_min_requires_args")
        return min(float(arg) for arg in args)
    if name == "table_max":
        if not args:
            raise ValueError("table_max_requires_args")
        return max(float(arg) for arg in args)
    raise ValueError(f"unsupported_operator:{name}")


def _evaluate_expression(expression: str, results: list[Any]) -> Any:
    """Evaluate one nested expression."""
    expression = expression.strip()
    if not expression:
        raise ValueError("empty_expression")

    if not re.match(r"^[a-zA-Z_]+\(", expression):
        return _parse_literal(expression, results)

    name, raw_args = expression.split("(", 1)
    name = name.strip()
    if name not in SUPPORTED_OPERATORS:
        raise ValueError(f"unsupported_operator:{name}")
    if not raw_args.endswith(")"):
        raise ValueError("malformed_expression")
    inner = raw_args[:-1]
    args = [_evaluate_expression(part, results) for part in split_top_level(inner)]
    return _apply_operator(name, args)


def execute_program(program: str) -> ExecutionResult:
    """Execute a FinQA-style program string and return the final value."""
    stripped = program.strip()
    if not stripped:
        return ExecutionResult(executable=False, value=None, error="empty_program")

    results: list[Any] = []
    try:
        for expression in split_top_level(stripped):
            results.append(_evaluate_expression(expression, results))
        if not results:
            return ExecutionResult(executable=False, value=None, error="no_expression")
        return ExecutionResult(executable=True, value=results[-1], error="")
    except Exception as exc:  # noqa: BLE001
        return ExecutionResult(executable=False, value=None, error=str(exc))

