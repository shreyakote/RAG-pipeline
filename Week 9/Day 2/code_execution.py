import ast

from crewai.tools import BaseTool


class CodeExecutionTool(BaseTool):
    name: str = "Code Calculator"
    description: str = (
        "Safely calculate basic mathematical expressions. "
        "Use this tool for numerical calculations."
    )

    def _run(self, expression: str) -> str:
        """Safely evaluate a basic mathematical expression."""

        try:
            tree = ast.parse(expression, mode="eval")

            allowed_nodes = (
                ast.Expression,
                ast.Constant,
                ast.BinOp,
                ast.UnaryOp,
                ast.Add,
                ast.Sub,
                ast.Mult,
                ast.Div,
                ast.Mod,
                ast.Pow,
                ast.USub,
                ast.UAdd,
            )

            for node in ast.walk(tree):
                if not isinstance(node, allowed_nodes):
                    return "Only basic mathematical expressions are allowed."

            result = eval(
                compile(tree, "<expression>", "eval"),
                {"__builtins__": {}},
                {},
            )

            return str(result)

        except (SyntaxError, ValueError, TypeError, ZeroDivisionError) as error:
            return f"Calculation failed: {error}"