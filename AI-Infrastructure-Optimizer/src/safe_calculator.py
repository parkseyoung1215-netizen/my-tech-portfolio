"""Bounded integer expression parser. No execution of Python source code."""

import ast
import re

MAX_LENGTH = 256
MAX_NODES = 64
MAX_DEPTH = 16
MAX_LITERAL = 999_999_999_999
MAX_RESULT_BITS = 128


class ToolDeclined(ValueError):
    def __init__(self, reason):
        self.reason = reason
        super().__init__(reason)


def parse_expression(text):
    """Admit complete ASCII integer +,-,* expressions only; no text extraction."""
    if not isinstance(text, str):
        raise ToolDeclined("not_text")
    if len(text) > MAX_LENGTH:
        raise ToolDeclined("length_limit")
    text = text.strip(" \t\r\n")
    if not text or re.fullmatch(r"[0-9+*()\- \t\r\n]+", text) is None:
        raise ToolDeclined("not_integer_expression")
    try:
        tree = ast.parse(text, mode="eval")
    except (SyntaxError, ValueError, RecursionError):
        raise ToolDeclined("invalid_syntax") from None
    if sum(1 for _ in ast.walk(tree)) > MAX_NODES:
        raise ToolDeclined("node_limit")
    operations = 0

    def validate(node, depth):
        nonlocal operations
        if depth > MAX_DEPTH:
            raise ToolDeclined("depth_limit")
        if isinstance(node, ast.Constant) and type(node.value) is int:
            if abs(node.value) > MAX_LITERAL:
                raise ToolDeclined("literal_limit")
        elif isinstance(node, ast.BinOp) and type(node.op) in (ast.Add, ast.Sub, ast.Mult):
            operations += 1
            validate(node.left, depth + 1)
            validate(node.right, depth + 1)
        elif isinstance(node, ast.UnaryOp) and type(node.op) in (ast.UAdd, ast.USub):
            validate(node.operand, depth + 1)
        else:
            raise ToolDeclined("unsupported_operation")

    validate(tree.body, 0)
    if operations == 0:
        raise ToolDeclined("no_binary_operation")
    return tree


def calculate(tree):
    """Walk admitted nodes with explicit operations and bounded intermediates."""
    def visit(node, depth=0):
        if depth > MAX_DEPTH:
            raise ToolDeclined("depth_limit")
        if isinstance(node, ast.Constant) and type(node.value) is int:
            if abs(node.value) > MAX_LITERAL:
                raise ToolDeclined("literal_limit")
            value = node.value
        elif isinstance(node, ast.UnaryOp) and type(node.op) in (ast.UAdd, ast.USub):
            value = visit(node.operand, depth + 1)
            if isinstance(node.op, ast.USub):
                value = -value
        elif isinstance(node, ast.BinOp) and type(node.op) in (ast.Add, ast.Sub, ast.Mult):
            left, right = visit(node.left, depth + 1), visit(node.right, depth + 1)
            if isinstance(node.op, ast.Add):
                value = left + right
            elif isinstance(node.op, ast.Sub):
                value = left - right
            else:
                value = left * right
        else:
            raise ToolDeclined("unsupported_operation")
        if value.bit_length() > MAX_RESULT_BITS:
            raise ToolDeclined("result_limit")
        return value
    if not isinstance(tree, ast.Expression):
        raise ToolDeclined("invalid_tree")
    return visit(tree.body)
