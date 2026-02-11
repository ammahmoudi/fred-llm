"""Tests for data augmentation strategies."""

from pathlib import Path

import pytest

from src.data.augmentation import DataAugmenter
from src.data.augmentations import (
    ApproximateOnlyAugmentation,
    BoundaryLayerAugmentation,
    CompactSupportAugmentation,
    DivergentKernelAugmentation,
    IllPosedAugmentation,
    MixedTypeAugmentation,
    NoSolutionAugmentation,
    OscillatorySolutionAugmentation,
    RangeViolationAugmentation,
    ResonanceAugmentation,
    WeaklySingularAugmentation,
)
from src.data.fredholm_loader import FredholmDatasetLoader


@pytest.fixture
def sample_dataset_path() -> Path:
    """Return path to sample dataset."""
    return Path("data/raw/Fredholm_Dataset_Sample.csv")


@pytest.fixture
def fredholm_loader(sample_dataset_path: Path) -> FredholmDatasetLoader:
    """Create FredholmDatasetLoader instance."""
    return FredholmDatasetLoader(data_path=sample_dataset_path, auto_download=False)


@pytest.fixture
def data_augmenter() -> DataAugmenter:
    """Create DataAugmenter instance."""
    return DataAugmenter()


class TestDataAugmentation:
    """Test data augmentation strategies."""

    @pytest.mark.skipif(
        not Path("data/raw/Fredholm_Dataset_Sample.csv").exists(),
        reason="Sample dataset not found",
    )
    def test_substitute_augmentation(
        self,
        sample_dataset_path: Path,
        fredholm_loader: FredholmDatasetLoader,
        data_augmenter: DataAugmenter,
    ) -> None:
        """Test variable substitution augmentation."""
        fredholm_loader.max_samples = 2
        equations = fredholm_loader.load()
        eq_dicts = [eq.to_dict() for eq in equations]
        augmenter = DataAugmenter(strategies=["substitute"])
        augmented = augmenter.augment(eq_dicts, multiplier=3)

        # Should have more equations
        assert len(augmented) > len(eq_dicts)

    @pytest.mark.skipif(
        not Path("data/raw/Fredholm_Dataset_Sample.csv").exists(),
        reason="Sample dataset not found",
    )
    def test_scale_augmentation(
        self,
        sample_dataset_path: Path,
        fredholm_loader: FredholmDatasetLoader,
        data_augmenter: DataAugmenter,
    ) -> None:
        """Test lambda coefficient scaling augmentation."""
        fredholm_loader.max_samples = 2
        equations = fredholm_loader.load()
        eq_dicts = [eq.to_dict() for eq in equations]
        augmenter = DataAugmenter(strategies=["scale"])
        augmented = augmenter.augment(eq_dicts, multiplier=2)

        assert len(augmented) > len(eq_dicts)

    @pytest.mark.skipif(
        not Path("data/raw/Fredholm_Dataset_Sample.csv").exists(),
        reason="Sample dataset not found",
    )
    def test_shift_augmentation(
        self,
        sample_dataset_path: Path,
        fredholm_loader: FredholmDatasetLoader,
        data_augmenter: DataAugmenter,
    ) -> None:
        """Test domain shift augmentation."""
        fredholm_loader.max_samples = 2
        equations = fredholm_loader.load()
        eq_dicts = [eq.to_dict() for eq in equations]
        augmenter = DataAugmenter(strategies=["shift"])
        augmented = augmenter.augment(eq_dicts, multiplier=2)

        assert len(augmented) > len(eq_dicts)

    @pytest.mark.skipif(
        not Path("data/raw/Fredholm_Dataset_Sample.csv").exists(),
        reason="Sample dataset not found",
    )
    def test_compose_augmentation(
        self,
        sample_dataset_path: Path,
        fredholm_loader: FredholmDatasetLoader,
        data_augmenter: DataAugmenter,
    ) -> None:
        """Test kernel composition augmentation."""
        fredholm_loader.max_samples = 2
        equations = fredholm_loader.load()
        eq_dicts = [eq.to_dict() for eq in equations]
        augmenter = DataAugmenter(strategies=["compose"])
        augmented = augmenter.augment(eq_dicts, multiplier=2)

        assert len(augmented) > len(eq_dicts)

    @pytest.mark.skipif(
        not Path("data/raw/Fredholm_Dataset_Sample.csv").exists(),
        reason="Sample dataset not found",
    )
    def test_combined_augmentation(
        self,
        sample_dataset_path: Path,
        fredholm_loader: FredholmDatasetLoader,
        data_augmenter: DataAugmenter,
    ) -> None:
        """Test combining multiple augmentation strategies."""
        fredholm_loader.max_samples = 3
        equations = fredholm_loader.load()
        eq_dicts = [eq.to_dict() for eq in equations]

        # Apply multiple strategies
        augmenter = DataAugmenter(strategies=["substitute", "scale"])
        augmented = augmenter.augment(eq_dicts, multiplier=3)

        # Should have significantly more equations
        expansion_ratio = len(augmented) / len(equations)
        assert expansion_ratio >= 2.0

    @pytest.mark.skipif(
        not Path("data/raw/Fredholm_Dataset_Sample.csv").exists(),
        reason="Sample dataset not found",
    )
    def test_augmentation_preserves_structure(
        self,
        sample_dataset_path: Path,
        fredholm_loader: FredholmDatasetLoader,
    ) -> None:
        """Test that all dataset entries (original and augmented) have unified schema."""
        fredholm_loader.max_samples = 1
        equations = fredholm_loader.load()
        eq_dicts = [eq.to_dict() for eq in equations]

        augmenter = DataAugmenter(strategies=["scale"])
        augmented = augmenter.augment(eq_dicts, multiplier=2)

        # Check core required fields are present in ALL entries (original + augmented)
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

        assert len(augmented) > 0, "No entries found"

        for entry in augmented:
            for field in core_fields:
                assert field in entry, f"Missing required field: {field}"

            # Verify correct types
            assert isinstance(entry["augmented"], bool)
            assert isinstance(entry["has_solution"], bool)
            assert isinstance(entry["recommended_methods"], list)
            assert entry["solution_type"] in [
                "exact_symbolic",
                "approx_coef",
                "discrete_points",
                "series",
                "family",
                "regularized",
                "none",
            ]
            assert entry["edge_case"] is None or isinstance(entry["edge_case"], str)

            # Verify augmentation tracking
            if entry["augmented"]:
                assert entry["augmentation_type"] != "original"
            else:
                assert entry["augmentation_type"] == "original"
                assert entry["augmentation_variant"] == "fredholm_dataset"

    def test_unified_schema_basic_augmentations(self) -> None:
        """Test that basic augmentations output unified 16-field schema."""
        from src.data.augmentations import (
            ComposeAugmentation,
            ScaleAugmentation,
            ShiftAugmentation,
            SubstituteAugmentation,
        )

        sample_eq = {
            "u": "x",
            "f": "x",
            "kernel": "x*t",
            "lambda": "1",
            "lambda_val": "1",
            "a": "0",
            "b": "1",
        }

        strategies = [
            SubstituteAugmentation(),
            ScaleAugmentation(),
            ShiftAugmentation(),
            ComposeAugmentation(),
        ]

        for strategy in strategies:
            results = strategy.augment(sample_eq)
            assert len(results) > 0, f"{strategy.strategy_name} produced no results"

            for result in results:
                # Check all 16 fields present
                assert result["has_solution"] is True
                assert result["solution_type"] == "exact_symbolic"
                assert result["edge_case"] is None
                assert isinstance(result["reason"], str)
                assert isinstance(result["recommended_methods"], list)
                assert result["recommended_methods"] == []
                assert result["numerical_challenge"] is None
                assert "augmentation_variant" in result


class TestEdgeCaseAugmentations:
    """Test edge case augmentation strategies."""

    def test_none_solution_augmentation(self) -> None:
        """Test no-solution (singular) case generation."""
        augmenter = NoSolutionAugmentation()

        # Create a sample equation
        sample_eq = {
            "u": "x",
            "f": "x",
            "kernel": "x*t",
            "lambda": "1",
            "lambda_val": "1",
            "a": "0",
            "b": "1",
        }

        # Generate no-solution cases
        no_sol_cases = augmenter.augment(sample_eq)

        # Should generate 3 cases (constant, separable, symmetric kernels)
        assert len(no_sol_cases) == 3

        # All should be marked as no solution
        for case in no_sol_cases:
            assert case["has_solution"] is False
            assert case["solution_type"] == "none"
            assert "reason" in case
            assert "Fredholm Alternative" in case["reason"]
            assert case["augmented"] is True
            # augmentation_type and edge_case should be the strategy name
            assert case["augmentation_type"] in [
                "eigenvalue_cases",
                "range_violation",
                "divergent_kernel",
                "disconnected_support",
            ]
            assert case["edge_case"] in [
                "eigenvalue_cases",
                "range_violation",
                "divergent_kernel",
                "disconnected_support",
            ]
            # Solution should be empty for no-solution cases
            assert case["u"] == ""

    def test_approximate_only_augmentation(self) -> None:
        """Test approximate-only (numerical) case generation."""
        augmenter = ApproximateOnlyAugmentation(num_sample_points=5)

        sample_eq = {
            "u": "x",
            "f": "x",
            "kernel": "x*t",
            "lambda": "0.5",
            "lambda_val": "0.5",
            "a": "0",
            "b": "1",
        }

        # Generate approximate-only cases
        approx_cases = augmenter.augment(sample_eq)

        # Should generate 3 cases (Gaussian, exponential, sinc kernels)
        assert len(approx_cases) == 3

        # All should require numerical methods
        for case in approx_cases:
            assert case["has_solution"] is True
            assert case["solution_type"] == "discrete_points"
            assert "numerical_method" in case
            assert "sample_points" in case
            assert "sample_values" in case
            assert len(case["sample_points"]) == 5
            assert len(case["sample_values"]) == 5
            assert case["augmented"] is True
            assert case["augmentation_type"] == "complex_kernels"
            assert case["edge_case"] == "complex_kernels"
            assert "reason" in case
            # Solution should be empty for numerical-only cases (no closed form)
            assert case["u"] == ""

    def test_ill_posed_augmentation(self) -> None:
        """Test ill-posed (1st kind) case generation."""
        augmenter = IllPosedAugmentation(num_sample_points=5, regularization_param=0.01)

        sample_eq = {
            "u": "x",
            "f": "x",
            "kernel": "x*t",
            "lambda": "1",
            "lambda_val": "1",
            "a": "0",
            "b": "1",
        }

        # Generate ill-posed cases
        ill_posed_cases = augmenter.augment(sample_eq)

        # Should generate 3 cases (simple, exponential, oscillatory)
        assert len(ill_posed_cases) == 3

        # All should be ill-posed first kind
        for case in ill_posed_cases:
            assert case["equation_type"] == "fredholm_first_kind"
            assert case["is_ill_posed"] is True
            assert case["requires_regularization"] is True
            assert "recommended_methods" in case
            assert len(case["recommended_methods"]) > 0
            assert case["solution_type"] == "regularized"
            assert case["augmented"] is True
            assert case["augmentation_type"] == "ill_posed"
            assert case["edge_case"] == "ill_posed"
            assert "reason" in case
            assert "warning" in case
            assert case["regularization_param"] == 0.01
            # Lambda should be 0 for first kind (no lambda parameter)
            assert case["lambda_val"] == "0"

    def test_edge_case_strategies_with_augmenter(self) -> None:
        """Test using edge case strategies through DataAugmenter."""
        sample_eq = {
            "u": "x",
            "f": "x",
            "kernel": "x*t",
            "lambda": "1",
            "lambda_val": "1",
            "a": "0",
            "b": "1",
        }

        # Test none_solution folder strategy (runs 4 strategies: eigenvalue_issue + range_violation + divergent_kernel + disconnected_support)
        augmenter = DataAugmenter(strategies=["none_solution"])
        augmented = augmenter.augment([sample_eq], multiplier=2)
        assert any(
            eq.get("edge_case")
            in [
                "eigenvalue_issue",
                "range_violation",
                "divergent_kernel",
                "disconnected_support",
            ]
            for eq in augmented
        )

        # Test approx_coef folder strategy (runs 5 strategies)
        augmenter = DataAugmenter(strategies=["approx_coef"])
        augmented = augmenter.augment([sample_eq], multiplier=2)
        assert any(
            eq.get("edge_case")
            in [
                "boundary_layer",
                "weakly_singular",
                "oscillatory_solution",
                "mixed_type",
                "compact_support",
            ]
            for eq in augmented
        )

        # Test regularized folder strategy (runs ill_posed)
        augmenter = DataAugmenter(strategies=["regularized"])
        augmented = augmenter.augment([sample_eq], multiplier=2)
        assert any(eq.get("edge_case") == "ill_posed" for eq in augmented)

    def test_combined_basic_and_edge_strategies(self) -> None:
        """Test combining basic and edge case strategies."""
        sample_eq = {
            "u": "x",
            "f": "x",
            "kernel": "x*t",
            "lambda": "1",
            "lambda_val": "1",
            "a": "0",
            "b": "1",
        }

        # Combine basic and folder-based edge case strategies
        augmenter = DataAugmenter(
            strategies=["substitute", "scale", "none", "approx_coef"]
        )
        # Use higher multiplier to ensure all strategies get applied
        augmented = augmenter.augment([sample_eq], multiplier=15)

        # Should have mix of regular and edge cases
        regular_cases = [eq for eq in augmented if not eq.get("edge_case")]
        edge_cases = [eq for eq in augmented if eq.get("edge_case")]

        assert len(regular_cases) > 0
        assert len(edge_cases) > 0
        assert len(augmented) > 1

    def test_none_solution_eigenvalue_detection(self) -> None:
        """Test that no-solution cases correctly identify eigenvalues."""
        augmenter = NoSolutionAugmentation()

        sample_eq = {
            "u": "x",
            "f": "x",
            "kernel": "1",
            "lambda": "1",
            "lambda_val": "1",
            "a": "0",
            "b": "1",
        }

        cases = augmenter.augment(sample_eq)

        # First case should be constant kernel with eigenvalue 1/(b-a) = 1
        constant_kernel_case = [
            c
            for c in cases
            if c.get("augmentation_variant") == "constant_kernel_eigenvalue"
        ][0]
        assert constant_kernel_case["kernel"] == "1"
        assert float(constant_kernel_case["lambda_val"]) == 1.0  # 1/(1-0)

    def test_approximate_only_sample_points(self) -> None:
        """Test that approximate-only cases generate valid sample points."""
        augmenter = ApproximateOnlyAugmentation(num_sample_points=10)

        sample_eq = {
            "u": "x",
            "f": "1",
            "kernel": "x*t",
            "lambda": "0.5",
            "lambda_val": "0.5",
            "a": "0",
            "b": "1",
        }

        cases = augmenter.augment(sample_eq)

        for case in cases:
            # Sample points should be within domain
            assert all(0 <= x <= 1 for x in case["sample_points"])
            # Sample values should be finite numbers
            assert all(abs(val) < 1e6 for val in case["sample_values"])
            # Should have correct number of points
            assert len(case["sample_points"]) == 10


class TestAdvancedEdgeCases:
    """Test advanced edge case augmentation strategies."""

    def test_weakly_singular_augmentation(self) -> None:
        """Test weakly singular kernel generation."""
        augmenter = WeaklySingularAugmentation(num_sample_points=15)

        sample_eq = {
            "u": "x",
            "f": "x",
            "kernel": "x*t",
            "lambda": "1",
            "a": "0",
            "b": "1",
        }

        cases = augmenter.augment(sample_eq)

        assert len(cases) == 3
        for case in cases:
            assert case["edge_case"] == "weakly_singular"
            assert case["augmentation_type"] == "weakly_singular"
            assert "singularity_type" in case
            assert case["singularity_type"] in [
                "logarithmic",
                "power_law",
                "algebraic_mixed",
            ]
            assert "singularity_order" in case

    def test_boundary_layer_augmentation(self) -> None:
        """Test boundary layer solution generation."""
        augmenter = BoundaryLayerAugmentation(epsilon=0.01, num_sample_points=20)

        sample_eq = {
            "u": "x",
            "f": "x",
            "kernel": "x*t",
            "lambda": "1",
            "a": "0",
            "b": "1",
        }

        cases = augmenter.augment(sample_eq)

        assert len(cases) == 3
        for case in cases:
            assert case["edge_case"] == "boundary_layer"
            assert case["augmentation_type"] == "boundary_layer"
            assert case["layer_location"] in ["left", "right", "both"]
            assert case["layer_width_estimate"] == 0.01
            assert case["gradient_scale"] == 100.0

    def test_resonance_augmentation(self) -> None:
        """Test resonance/critical point generation."""
        augmenter = ResonanceAugmentation(perturbation=0.001)

        sample_eq = {
            "u": "x",
            "f": "x",
            "kernel": "x*t",
            "lambda": "1",
            "a": "0",
            "b": "1",
        }

        cases = augmenter.augment(sample_eq)

        assert len(cases) >= 2, f"Expected at least 2 cases, got {len(cases)}"
        for case in cases:
            assert case["augmentation_type"] == "resonance"
            if "is_critical" in case and case["is_critical"]:
                assert case["edge_case"] == "resonance"
                assert "eigenvalue_approximate" in case
                assert "solution_multiplicity" in case
            else:
                assert case["edge_case"] == "near_resonance"

    def test_range_violation_augmentation(self) -> None:
        """Test range space violation generation."""
        augmenter = RangeViolationAugmentation()

        sample_eq = {
            "u": "x",
            "f": "x",
            "kernel": "x*t",
            "lambda": "1",
            "a": "0",
            "b": "1",
        }

        cases = augmenter.augment(sample_eq)

        assert len(cases) == 3
        for case in cases:
            assert case["edge_case"] == "range_violation"
            assert case["has_solution"] is False
            assert case["solution_type"] == "none"
            assert "operator_property" in case
            assert case["operator_property"] in [
                "even_symmetry",
                "separable_rank_one",
                "finite_rank",
            ]

    def test_divergent_kernel_augmentation(self) -> None:
        """Test non-integrable singularity generation."""
        augmenter = DivergentKernelAugmentation()

        sample_eq = {
            "u": "x",
            "f": "x",
            "kernel": "x*t",
            "lambda": "1",
            "a": "0",
            "b": "1",
        }

        cases = augmenter.augment(sample_eq)

        assert len(cases) == 3
        for case in cases:
            assert case["edge_case"] == "divergent_kernel"
            assert case["has_solution"] is False
            assert "singularity_order" in case
            assert case["singularity_order"] >= 1.0
            assert "divergence_type" in case

    def test_mixed_type_augmentation(self) -> None:
        """Test Volterra-Fredholm mixed type generation."""
        augmenter = MixedTypeAugmentation()

        sample_eq = {
            "u": "x",
            "f": "x",
            "kernel": "x*t",
            "lambda": "1",
            "a": "0",
            "b": "1",
        }

        cases = augmenter.augment(sample_eq)

        assert len(cases) == 3
        for case in cases:
            assert case["edge_case"] == "mixed_type"
            assert case["has_solution"] is True
            assert case["solution_type"] == "approx_coef"
            assert "causal_structure" in case
            assert case["causal_structure"] in ["partial", "approximate", "explicit"]

    def test_oscillatory_solution_augmentation(self) -> None:
        """Test rapidly oscillating solution generation."""
        augmenter = OscillatorySolutionAugmentation(
            base_frequency=10.0, num_sample_points=100
        )

        sample_eq = {
            "u": "x",
            "f": "x",
            "kernel": "x*t",
            "lambda": "1",
            "a": "0",
            "b": "1",
        }

        cases = augmenter.augment(sample_eq)

        assert len(cases) == 3
        for case in cases:
            assert case["edge_case"] == "oscillatory_solution"
            assert case["has_solution"] is True
            # Check frequency info (varies by variant)
            assert "oscillation_frequency" in case or "frequencies" in case
            assert "nyquist_samples_required" in case

    def test_compact_support_augmentation(self) -> None:
        """Test compact support kernel generation."""
        augmenter = CompactSupportAugmentation(bandwidth=0.1)

        sample_eq = {
            "u": "x",
            "f": "x",
            "kernel": "x*t",
            "lambda": "1",
            "a": "0",
            "b": "1",
        }

        cases = augmenter.augment(sample_eq)

        assert len(cases) >= 2, f"Expected at least 2 cases, got {len(cases)}"
        for case in cases:
            assert case["edge_case"] == "compact_support"
            assert "support_type" in case
            assert case["support_type"] in [
                "band",
                "localized_box",
                "disconnected_regions",
            ]
            assert "zero_fraction" in case
            assert 0 < case["zero_fraction"] < 1
