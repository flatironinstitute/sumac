import pytest
import torch

from sumac import SumacConfig, sumac_factorize


def valid_inputs():
    return {
        "S_index": torch.tensor([[0, 1], [0, 1]], dtype=torch.int64),
        "S_value": torch.tensor([1.0, 2.0]),
        "shape": (2, 2),
        "config": SumacConfig(verbose=False),
    }


@pytest.mark.parametrize(
    ("replacement", "exception", "match"),
    [
        ({"S_index": torch.tensor([0, 1])}, ValueError, "shape"),
        ({"S_index": torch.tensor([[0, 1], [0, 1]], dtype=torch.float32)}, TypeError, "index dtype"),
        ({"S_value": torch.tensor([[1.0, 2.0]])}, ValueError, "S_value must have shape"),
        ({"S_value": torch.tensor([1.0])}, ValueError, "nnz dimensions"),
        ({"S_value": torch.tensor([1.0, float("inf")])}, ValueError, "finite"),
        ({"S_value": torch.tensor([1.0, -2.0])}, ValueError, "nonnegative"),
        ({"S_value": torch.tensor([1.0, 0.0])}, ValueError, "nonzero COO"),
        ({"shape": (2, 0)}, ValueError, "positive integers"),
        ({"S_index": torch.tensor([[0, 2], [0, 1]])}, ValueError, "row indices"),
        ({"S_index": torch.tensor([[0, 0], [0, 0]])}, ValueError, "duplicate"),
    ],
)
def test_invalid_sparse_inputs(replacement, exception, match):
    arguments = valid_inputs()
    arguments.update(replacement)

    with pytest.raises(exception, match=match):
        sumac_factorize(**arguments)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("rank", 0, "rank"),
        ("max_iterations", -1, "max_iterations"),
        ("num_blocks", 0, "num_blocks"),
        ("cache_mb", float("nan"), "cache_mb"),
        ("dtype", torch.float16, "dtype"),
        ("eval_interval", 0, "eval_interval"),
        ("batch_blocks", 0, "batch_blocks"),
        ("learning_rate", -0.1, "learning_rate"),
        ("momentum", 1.0, "momentum"),
    ],
)
def test_invalid_config_fields(field, value, match):
    arguments = valid_inputs()
    setattr(arguments["config"], field, value)

    with pytest.raises((TypeError, ValueError), match=match):
        sumac_factorize(**arguments)


def test_float64_and_tf32_are_mutually_exclusive():
    arguments = valid_inputs()
    arguments["config"].dtype = torch.float64
    arguments["config"].allow_tf32 = True

    with pytest.raises(ValueError, match="mutually exclusive"):
        sumac_factorize(**arguments)


def test_initial_factors_must_be_provided_together():
    arguments = valid_inputs()
    arguments["A_init"] = torch.ones(2, arguments["config"].rank)

    with pytest.raises(ValueError, match="both be provided"):
        sumac_factorize(**arguments)


def test_initial_factor_shapes_are_validated():
    arguments = valid_inputs()
    arguments["A_init"] = torch.ones(1, arguments["config"].rank)
    arguments["B_init"] = torch.ones(2, arguments["config"].rank)

    with pytest.raises(ValueError, match="A_init must have shape"):
        sumac_factorize(**arguments)


def test_num_blocks_is_limited_by_effective_shape():
    arguments = valid_inputs()
    arguments["S_index"] = torch.tensor([[0], [0]])
    arguments["S_value"] = torch.tensor([1.0])
    arguments["config"].num_blocks = 2

    with pytest.raises(ValueError, match="effective shape"):
        sumac_factorize(**arguments)
