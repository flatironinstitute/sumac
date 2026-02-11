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
    Simple, fixed minibatches (only randomize over the batch order)
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
        start, end = block_span(block_id, self.m, self.num_blocks)
        indices = torch.arange(start, end, dtype=torch.long)
        return block_id, self.block_edge_idx[int(block_id)], indices

class StochasticRowBlockDataset(Dataset):
    """
    Stochastic version of RowBlockDataset
    Re-partitions rows into blocks every time .reshuffle() is called.
    """
    def __init__(self, S_index: torch.LongTensor, S_value: torch.Tensor,
                 m: int, num_blocks: int):
        self.S_index = S_index
        self.S_value = S_value
        self.m = m
        self.num_blocks = num_blocks
        
        #This takes a minute to build and is not used - disabling it until this is needed
        # # 1. Build row-to-edge index map once (CPU)
        # # This allows us to quickly find all non-zeros for a given set of rows
        # rows = self.S_index[0].detach().cpu()
        # self.row_to_edges = [[] for _ in range(m)]
        # for edge_idx, row_idx in enumerate(rows):
        #     self.row_to_edges[row_idx.item()].append(edge_idx)
        
        # # Convert to tensors for faster concatenation
        # self.row_to_edges = [torch.tensor(e, dtype=torch.long) for e in self.row_to_edges]
        
        # Initial partition
        self.reshuffle()

    def reshuffle(self):
        """
        Vectorized reshuffle: Re-partitions rows into blocks in a single go.
        """
        # 1. Randomly permute rows
        perm = torch.randperm(self.m)
        perm_inv = torch.empty_like(perm)
        perm_inv[perm] = torch.arange(self.m)
        
        # 2. Determine block for each edge based on its row's position in perm
        block_size = (self.m + self.num_blocks - 1) // self.num_blocks
        
        # We do this calculation on CPU to keep dataset memory light
        edge_rows = self.S_index[0].detach().cpu()
        edge_block_ids = perm_inv[edge_rows] // block_size
        
        # 3. Sort edge indices by their block_id
        sorted_edge_indices = torch.argsort(edge_block_ids)
        
        # 4. Find boundaries and slice
        counts = torch.bincount(edge_block_ids, minlength=self.num_blocks)
        offsets = torch.zeros(self.num_blocks + 1, dtype=torch.long)
        torch.cumsum(counts, dim=0, out=offsets[1:])
        
        self.block_edge_idx = []
        self.block_row_indices = []
        for b in range(self.num_blocks):
            self.block_edge_idx.append(sorted_edge_indices[offsets[b]:offsets[b+1]])
            self.block_row_indices.append(perm[b*block_size : min(self.m, (b+1)*block_size)])

    def __len__(self):
        return self.num_blocks

    def __getitem__(self, block_id):
        bid = int(block_id)
        return bid, self.block_edge_idx[bid], self.block_row_indices[bid]