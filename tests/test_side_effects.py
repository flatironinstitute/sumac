"""Tests that SUMAC does not alter state owned by its caller."""

import importlib
import random

import pytest
import torch

from sumac import SumacConfig, SumacMethod, sumac_factorize
from sumac.training import gd
from sumac.training import salsa


def test_config_construction_does_not_modify_python_rng_state():
    original_state = random.getstate()
    try:
        random.seed(1729)
        state_before = random.getstate()

        SumacConfig(seed=42, verbose=False)

        assert random.getstate() == state_before
    finally:
        random.setstate(original_state)


@pytest.mark.parametrize("seed", [42, None])
def test_get_generator_does_not_modify_torch_rng_state(seed):
    original_state = torch.random.get_rng_state()
    try:
        torch.manual_seed(1729)
        state_before = torch.random.get_rng_state().clone()
        config = SumacConfig(
            method=SumacMethod.SALSA,
            seed=seed,
            device=torch.device("cpu"),
            verbose=False,
        )

        generators = config.get_generator()

        torch.testing.assert_close(torch.random.get_rng_state(), state_before)
        initial_seeds = tuple(generator.initial_seed() for generator in generators)
        if seed is None:
            assert len(set(initial_seeds)) == len(initial_seeds)
        else:
            assert initial_seeds == (seed, seed + 1, seed + 2)
    finally:
        torch.random.set_rng_state(original_state)


def test_gd_block_shuffle_does_not_consume_global_torch_rng(monkeypatch):
    def fake_block_loss(kernel, A, B, *args, **kwargs):
        loss = (A.sum() + B.sum()).square()
        zero = loss.detach().new_zeros(())
        return loss, zero, zero, loss

    monkeypatch.setattr(gd, "block_loss_and_pred", fake_block_loss)
    monkeypatch.setattr(gd, "eval", lambda *args, **kwargs: (0.0, 0.0, 0.0))
    config = SumacConfig(
        method=SumacMethod.GD,
        rank=1,
        max_iterations=1,
        num_blocks=2,
        seed=42,
        device=torch.device("cpu"),
        shuffle_blocks=True,
        verbose=False,
    )
    S_index = torch.tensor([[0, 1], [0, 1]], dtype=torch.int64)
    S_value = torch.tensor([1.0, 1.0])
    original_state = torch.random.get_rng_state()
    try:
        torch.manual_seed(1729)
        state_before = torch.random.get_rng_state().clone()

        gd.GD_loop(S_index, S_value, 2, 2, config)

        torch.testing.assert_close(torch.random.get_rng_state(), state_before)
    finally:
        torch.random.set_rng_state(original_state)


def test_factorize_restores_matmul_precision(monkeypatch):
    sumac_module = importlib.import_module("sumac.sumac")
    precision_before = torch.get_float32_matmul_precision()
    requested_precision = "high" if precision_before != "high" else "highest"

    def fake_gd_loop(S_index, S_value, m, n, config, A_init, B_init):
        assert torch.get_float32_matmul_precision() == requested_precision
        return (
            torch.ones((m, config.rank), dtype=config.dtype),
            torch.ones((n, config.rank), dtype=config.dtype),
            [],
        )

    monkeypatch.setattr(sumac_module, "GD_loop", fake_gd_loop)
    config = SumacConfig(
        method=SumacMethod.GD,
        rank=1,
        max_iterations=0,
        num_blocks=1,
        allow_tf32=requested_precision == "high",
        device=torch.device("cpu"),
        verbose=False,
    )

    try:
        sumac_factorize(
            S_index=torch.tensor([[0], [0]], dtype=torch.int64),
            S_value=torch.tensor([1.0]),
            shape=(1, 1),
            config=config,
        )

        assert torch.get_float32_matmul_precision() == precision_before
    finally:
        torch.set_float32_matmul_precision(precision_before)


@pytest.mark.parametrize("method", [SumacMethod.GD, SumacMethod.SALSA])
def test_factorize_does_not_modify_input_tensors(monkeypatch, method):
    def fake_block_loss(kernel, A, B, *args, **kwargs):
        loss = (A.sum() + B.sum()).square()
        zero = loss.detach().new_zeros(())
        return loss, zero, zero, loss

    class ReferenceReluBatC:
        def resolve_params(self, *args, **kwargs):
            return {}

        def __call__(self, A, B, C):
            return torch.relu(B @ A.T) @ C

    monkeypatch.setattr(gd, "block_loss_and_pred", fake_block_loss)
    monkeypatch.setattr(gd, "eval", lambda *args, **kwargs: (0.0, 0.0, 0.0))
    monkeypatch.setattr(salsa, "configure_kernel_prec", lambda **kwargs: None)
    monkeypatch.setattr(salsa, "relu_bat_c_tuned", ReferenceReluBatC())
    monkeypatch.setattr(salsa, "eval", lambda *args, **kwargs: (0.0, 0.0, 0.0))

    S_index = torch.tensor([[0, 1], [0, 1]], dtype=torch.int64)
    S_value = torch.tensor([1.0, 2.0])
    A_init = torch.tensor([[1.0], [2.0]])
    B_init = torch.tensor([[3.0], [4.0]])
    inputs_before = tuple(
        tensor.clone() for tensor in (S_index, S_value, A_init, B_init)
    )
    config = SumacConfig(
        method=method,
        rank=1,
        max_iterations=1,
        num_blocks=1,
        eval_interval=1,
        learning_rate=0.1,
        device=torch.device("cpu"),
        verbose=False,
    )

    A_result, B_result, _ = sumac_factorize(
        S_index=S_index,
        S_value=S_value,
        shape=(2, 2),
        A_init=A_init,
        B_init=B_init,
        config=config,
    )

    assert not torch.equal(A_result, A_init)
    assert not torch.equal(B_result, B_init)
    for tensor, expected in zip(
        (S_index, S_value, A_init, B_init), inputs_before, strict=True
    ):
        torch.testing.assert_close(tensor, expected)
