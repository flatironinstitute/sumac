import torch
from torch.utils.data import Dataset


def collate_blocks(batch):
    return batch


class StochasticRowBlockDataset(Dataset):
    """
    Row blocks that are reshuffled by randomly repartitioning rows.
    """

    S_index: torch.Tensor
    S_value: torch.Tensor
    m: int
    num_blocks: int
    block_edge_idx: list[torch.Tensor]
    block_row_indices: list[torch.Tensor]
    gen: torch.Generator

    def __init__(
        self,
        S_index: torch.Tensor,
        S_value: torch.Tensor,
        m: int,
        num_blocks: int,
        gen: torch.Generator,
    ):
        self.S_index = S_index
        self.S_value = S_value
        self.m = m
        self.num_blocks = num_blocks
        self.gen = gen
        self.reshuffle()

    def reshuffle(self):
        dev = self.S_value.device
        perm = torch.randperm(self.m, device=dev, generator=self.gen)
        perm_inv = torch.empty_like(perm)
        perm_inv[perm] = torch.arange(self.m, device=dev)

        block_size = (self.m + self.num_blocks - 1) // self.num_blocks
        edge_rows = self.S_index[0]
        edge_block_ids = perm_inv[edge_rows] // block_size
        sorted_edge_indices = torch.argsort(edge_block_ids)

        counts = torch.bincount(edge_block_ids, minlength=self.num_blocks)
        offsets = torch.zeros(self.num_blocks + 1, dtype=torch.long, device=dev)
        torch.cumsum(counts, dim=0, out=offsets[1:])

        self.block_edge_idx = []
        self.block_row_indices = []
        for block_id in range(self.num_blocks):
            self.block_edge_idx.append(sorted_edge_indices[offsets[block_id]:offsets[block_id + 1]])
            self.block_row_indices.append(
                perm[block_id * block_size:min(self.m, (block_id + 1) * block_size)]
            )

    def __len__(self):
        return self.num_blocks

    def __getitem__(self, block_id):
        block_id = int(block_id)
        return block_id, self.block_edge_idx[block_id], self.block_row_indices[block_id]
