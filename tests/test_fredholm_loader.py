"""Tests for the Fredholm dataset loader."""

from src.data.fredholm_loader import (
    ExpressionType,
    FredholmEquation,
    _infer_expression_type,
)


class TestFredholmEquation:
    """Tests for FredholmEquation dataclass."""

    def test_from_csv_row_basic(self) -> None:
        """Test creating equation from CSV row with basic fields."""
        row = {
            "u": "x**2",
            "f": "x**2 + 2*x",
            "kernel": "x*t",
            "lambda": "0.5",
            "a": "0",
            "b": "1",
        }
        eq = FredholmEquation.from_csv_row(row)

        assert eq.u == "x**2"
        assert eq.f == "x**2 + 2*x"
        assert eq.kernel == "x*t"
        assert eq.lambda_val == "0.5"
        assert eq.a == "0"
        assert eq.b == "1"

    def test_from_csv_row_infers_types(self) -> None:
        """Test that expression types are inferred from expressions."""
        row = {
            "u": "sin(x)",
            "f": "cos(x) + exp(x)",
            "kernel": "cosh(x*t)",
            "lambda": "1.0",
            "a": "0",
            "b": "1",
        }
        eq = FredholmEquation.from_csv_row(row)

        assert eq.metadata["u_type"] == ExpressionType.TRIGONOMETRIC
        assert eq.metadata["f_type"] == ExpressionType.EXPONENTIAL
        assert eq.metadata["kernel_type"] == ExpressionType.HYPERBOLIC

    def test_to_dict(self) -> None:
        """Test converting equation to dictionary with unified schema."""
        eq = FredholmEquation(
            u="x",
            f="x + 1",
            kernel="x*t",
            lambda_val="0.5",
            a="0",
            b="1",
        )
        result = eq.to_dict()

        # Check core required fields are present
        core_fields = [
            "u",
            "f",
            "kernel",
            "lambda_val",
            "a",
            "b",
            "augmented",
            "augmentation_type",
            "augmentation_variant",
            "has_solution",
            "solution_type",
            "edge_case",
            "reason",
            "recommended_methods",
            "numerical_challenge",
        ]
        for field in core_fields:
            assert field in result, f"Missing required field: {field}"

        # Check core equation fields
        assert result["u"] == "x"
        assert result["f"] == "x + 1"
        assert result["kernel"] == "x*t"
        assert result["lambda_val"] == "0.5"
        assert result["a"] == "0"
        assert result["b"] == "1"

        # Check schema fields for original dataset
        assert result["augmented"] is False
        assert result["augmentation_type"] == "original"
        assert result["augmentation_variant"] == "fredholm_dataset"
        assert result["has_solution"] is True
        assert result["solution_type"] == "exact_symbolic"
        assert result["edge_case"] is None
        assert "Fredholm-LLM dataset" in result["reason"]
        assert isinstance(result["recommended_methods"], list)
        assert result["numerical_challenge"] is None

    def test_to_equation_string_symbolic(self) -> None:
        """Test symbolic equation string format."""
        eq = FredholmEquation(
            u="x",
            f="x + 1",
            kernel="x*t",
            lambda_val="0.5",
            a="0",
            b="1",
        )
        result = eq.to_equation_string(style="symbolic")

        assert "u(x)" in result
        assert "0.5" in result
        assert "x*t" in result

    def test_to_equation_string_latex(self) -> None:
        """Test LaTeX equation string format."""
        eq = FredholmEquation(
            u="x",
            f="x + 1",
            kernel="x*t",
            lambda_val="0.5",
            a="0",
            b="1",
        )
        result = eq.to_equation_string(style="latex")

        assert "\\int" in result
        assert "{0}" in result
        assert "{1}" in result

    def test_solution_property(self) -> None:
        """Test solution property returns u."""
        eq = FredholmEquation(
            u="x**2 + 1",
            f="x + 1",
            kernel="x*t",
            lambda_val="0.5",
            a="0",
            b="1",
        )
        assert eq.solution == "x**2 + 1"


class TestExpressionTypeInference:
    """Tests for expression type inference."""

    def test_infer_polynomial(self) -> None:
        """Test polynomial expression detection."""
        assert _infer_expression_type("x**2") == ExpressionType.POLYNOMIAL
        assert _infer_expression_type("x + 1") == ExpressionType.POLYNOMIAL
        assert _infer_expression_type("t**3 - 2*t") == ExpressionType.POLYNOMIAL

    def test_infer_trigonometric(self) -> None:
        """Test trigonometric expression detection."""
        assert _infer_expression_type("sin(x)") == ExpressionType.TRIGONOMETRIC
        assert _infer_expression_type("cos(t)") == ExpressionType.TRIGONOMETRIC
        assert _infer_expression_type("tan(x) + 1") == ExpressionType.TRIGONOMETRIC

    def test_infer_hyperbolic(self) -> None:
        """Test hyperbolic expression detection."""
        assert _infer_expression_type("sinh(x)") == ExpressionType.HYPERBOLIC
        assert _infer_expression_type("cosh(t)") == ExpressionType.HYPERBOLIC
        assert _infer_expression_type("tanh(x*t)") == ExpressionType.HYPERBOLIC

    def test_infer_exponential(self) -> None:
        """Test exponential expression detection."""
        assert _infer_expression_type("exp(x)") == ExpressionType.EXPONENTIAL
        assert _infer_expression_type("exp(-t**2)") == ExpressionType.EXPONENTIAL

    def test_infer_real_value(self) -> None:
        """Test constant/real value detection."""
        assert _infer_expression_type("5") == ExpressionType.REAL_VALUE
        assert _infer_expression_type("3.14159") == ExpressionType.REAL_VALUE
        assert _infer_expression_type("1 + 2") == ExpressionType.REAL_VALUE

    def test_exponential_takes_precedence(self) -> None:
        """Test that exp takes precedence over other types."""
        # Expression with both trig and exp should be classified as exp
        assert _infer_expression_type("exp(x) + sin(x)") == ExpressionType.EXPONENTIAL


class TestExpressionType:
    """Tests for ExpressionType enum."""

    def test_enum_values(self) -> None:
        """Test enum has expected values."""
        assert ExpressionType.REAL_VALUE.value == "real_value"
        assert ExpressionType.POLYNOMIAL.value == "polynomial"
        assert ExpressionType.TRIGONOMETRIC.value == "trigonometric"
        assert ExpressionType.HYPERBOLIC.value == "hyperbolic"
        assert ExpressionType.EXPONENTIAL.value == "exponential"
