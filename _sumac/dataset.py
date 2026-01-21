import torch
from torch.utils.data import Dataset, DataLoader, Subset

# ---------- helpers ----------
def block_span(block_id: int, m: int, num_blocks: int):
    block_size = (m + num_blocks - 1) // num_blocks  # ceil(n/num_blocks)
    start = block_id * block_size
    end = min(m, start + block_size)
    return start, end

# batch multiple blocks per step
def collate_blocks(batch):  # list[(block_id, edge_idx), ...]
    return batch

# ---------- dataset over row blocks ----------
class RowBlockDataset(Dataset):
    """
    Iterates over num_blocks. Each item returns (block_id, edge_idx)
    where edge_idx selects edges with row in this block's span.
    Expects S_index shape == (2, E), S_value shape == (E,)
    """
    def __init__(self, S_index: torch.LongTensor, S_value: torch.Tensor,
                 m: int, num_blocks: int):
        assert S_index.dim() == 2 and S_index.size(0) == 2, "S_index must be (2,E)"
        assert S_value.dim() == 1 and S_value.size(0) == S_index.size(1), "size mismatch"
        assert num_blocks >= 1, "num_blocks must be >= 1"
        self.S_index = S_index
        self.S_value = S_value
        self.m = m
        self.num_blocks = num_blocks
        self._rows = self.S_index[0]

        #precompute block edge partition
        self.block_edge_idx = []
        for block_id in range(self.num_blocks):
            start, end = block_span(block_id, self.m, self.num_blocks)
            mask = (self._rows >= start) & (self._rows < end)
            edge_idx = mask.nonzero(as_tuple=False).flatten()
            self.block_edge_idx.append(edge_idx)   # CPU LongTensor

    def __len__(self):
        return self.num_blocks

    def __getitem__(self, block_id):
        # start, end = block_span(block_id, self.m, self.num_blocks)
        # mask = (self._rows >= start) & (self._rows < end)
        # edge_idx = mask.nonzero(as_tuple=False).flatten()
        #return block_id, edge_idx
        return block_id, self.block_edge_idx[int(block_id)]
