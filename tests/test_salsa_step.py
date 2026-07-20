import torch

from sumac.training import salsa


class ReferenceReluBatC:
    def __call__(self, A, B, C):
        return torch.relu(B @ A.T) @ C


def reference_update(A, B, delta_B, edge_i, edge_j, values, momentum, unbias, learning_rate):
    pinvAt = torch.linalg.solve(A.T @ A, A.T).T
    latent = B @ A.T
    target = latent - torch.relu(latent)
    for value, row, column in zip(values, edge_i, edge_j, strict=True):
        target[column, row] = value
    least_squares_B = target @ pinvAt
    expected_delta = (
        (least_squares_B - B) * (1.0 - momentum) + delta_B * momentum
    )
    expected_B = B + learning_rate * expected_delta / unbias
    return expected_B, expected_delta


def test_single_salsa_factor_update_matches_reference(monkeypatch):
    monkeypatch.setattr(salsa, "relu_bat_c_tuned", ReferenceReluBatC())

    A = torch.tensor(
        [[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]], dtype=torch.float64
    )
    B = torch.tensor(
        [[0.5, -1.0], [2.0, 0.5], [-0.5, 1.5]], dtype=torch.float64
    )
    pinvAt = torch.linalg.solve(A.T @ A, A.T).T
    delta_B = torch.tensor(
        [[0.1, -0.2], [0.0, 0.3], [-0.1, 0.0]], dtype=torch.float64
    )
    edge_i = torch.tensor([0, 2, 1, 2])
    edge_j = torch.tensor([0, 0, 1, 2])
    values = torch.tensor([1.5, 0.25, 2.0, 3.0], dtype=torch.float64)
    momentum = torch.tensor(0.4, dtype=torch.float64)
    unbias = torch.tensor(1.0 - 0.4**3, dtype=torch.float64)
    learning_rate = torch.tensor(0.2, dtype=torch.float64)

    result_B, result_delta = salsa.lsq_update_single_gpu(
        Ar_dev=A,
        B_blk_dev=B,
        pinvAt_dev=pinvAt,
        dB_blk_dev=delta_B,
        edge_i=edge_i,
        edge_j=edge_j,
        blk_vals=values,
        momentum=momentum,
        unbias=unbias,
        lrate=learning_rate,
    )

    expected_B, expected_delta = reference_update(
        A,
        B,
        delta_B,
        edge_i,
        edge_j,
        values,
        momentum,
        unbias,
        learning_rate,
    )

    torch.testing.assert_close(result_delta, expected_delta)
    torch.testing.assert_close(result_B, expected_B)


def test_batch_salsa_update_maps_shuffled_rows_to_local_indices(monkeypatch):
    monkeypatch.setattr(salsa, "relu_bat_c_tuned", ReferenceReluBatC())
    batch_update = getattr(
        salsa.batch_update_single_gpu,
        "_torchdynamo_orig_callable",
        salsa.batch_update_single_gpu,
    )

    factor_fixed = torch.tensor(
        [[1.0, 0.0], [0.0, 2.0], [1.0, 1.0], [2.0, -1.0]],
        dtype=torch.float64,
    )
    row_indices = torch.tensor([3, 0, 2])
    B = torch.tensor(
        [[0.5, -1.0], [2.0, 0.5], [-0.5, 1.5]], dtype=torch.float64
    )
    delta_B = torch.tensor(
        [[0.1, -0.2], [0.0, 0.3], [-0.1, 0.0]], dtype=torch.float64
    )
    S_index = torch.tensor(
        [[3, 0, 2, 1, 3], [0, 2, 1, 1, 2]], dtype=torch.int64
    )
    S_value = torch.tensor([1.5, 0.25, 2.0, 9.0, 3.0], dtype=torch.float64)
    edge_idx = torch.tensor([0, 1, 2, 4])
    momentum = torch.tensor(0.4, dtype=torch.float64)
    unbias = torch.tensor(1.0 - 0.4**3, dtype=torch.float64)
    learning_rate = torch.tensor(0.2, dtype=torch.float64)

    result_B, result_delta = batch_update(
        S_idx_full=S_index,
        S_val_full=S_value,
        edge_idx=edge_idx,
        Factor_fixed=factor_fixed,
        row_indices=row_indices,
        B=B,
        dB=delta_B,
        momentum=momentum,
        unbias=unbias,
        lrate=learning_rate,
        m_fixed=factor_fixed.shape[0],
    )

    local_rows = torch.tensor([0, 1, 2, 0])
    output_rows = S_index[1, edge_idx]
    expected_B, expected_delta = reference_update(
        factor_fixed[row_indices],
        B,
        delta_B,
        local_rows,
        output_rows,
        S_value[edge_idx],
        momentum,
        unbias,
        learning_rate,
    )

    torch.testing.assert_close(result_delta, expected_delta)
    torch.testing.assert_close(result_B, expected_B)
