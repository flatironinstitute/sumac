import torch
from torch.utils.data import Dataset, DataLoader, Subset
from typing import List, Tuple, Any

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
    def __init__(self, S_index: torch.Tensor, S_value: torch.Tensor,
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
            self.block_edge_idx.append(edge_idx)   # CPU

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

    S_index: torch.Tensor
    S_value: torch.Tensor
    m: int
    num_blocks: int
    row_to_edges: list[torch.Tensor]
    block_edge_idx: list[torch.Tensor]
    block_row_indices: list[torch.Tensor]
    gen: torch.Generator

    def __init__(self,
            S_index: torch.Tensor,
            S_value: torch.Tensor,
            m: int,
            num_blocks: int,
            gen: torch.Generator
        ):

        ## TODO: consider deleting the force-cpu of S_index, S_value.
        self.S_index = S_index.detach().cpu() #NEW: force CPU
        self.S_value = S_value.detach().cpu() #NEW: force CPU
        self.m = m
        self.num_blocks = num_blocks
        self.gen = gen

        ## TODO: Note this check and the row-edge map (balance of this fn)
        # is not present in DBollweg branch        
        if torch.is_floating_point(self.S_index):
            raise ValueError("Index tensor is expected to be integer-valued.")


        # 1. Build row-to-edge index map once (CPU)
        # This allows us to quickly find all non-zeros for a given set of rows
        rows = self.S_index[0].detach().cpu()

        _row_to_edges = [[] for _ in range(m)]
        for edge_idx, row_idx in enumerate(rows):
            # TODO: check if this is assured
            # (call to .item() will throw if row_idx.numel() > 1)
            row_idx_scalar: int = int(row_idx.item())
            _row_to_edges[row_idx_scalar].append(edge_idx)
        
        # Convert to tensors for faster concatenation
        self.row_to_edges = [torch.tensor(e, dtype=torch.long) for e in _row_to_edges]
        
        # Initial partition
        self.reshuffle()


    def reshuffle(self):
        """
        Vectorized reshuffle: Re-partitions rows into blocks in a single go.
        """
        # 1. Randomly permute rows
        dev = self.S_value.device # TODO: confirm whether to instead do torch.device("cpu")
        perm = torch.randperm(self.m, device=dev, generator=self.gen)
        perm_inv = torch.empty_like(perm)
        perm_inv[perm] = torch.arange(self.m, device=dev)
        
        # 2. Determine block for each edge based on its row's position in perm
        block_size = (self.m + self.num_blocks - 1) // self.num_blocks
        
        edge_rows = self.S_index[0]
        edge_block_ids = perm_inv[edge_rows] // block_size
        
        # 3. Sort edge indices by their block_id
        sorted_edge_indices = torch.argsort(edge_block_ids)
        
        # 4. Find boundaries and slice
        counts = torch.bincount(edge_block_ids, minlength=self.num_blocks)
        offsets = torch.zeros(self.num_blocks + 1, dtype=torch.long, device=dev)
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


class MultiGPUStochasticRowBlockDataset(Dataset):
    """
    Minimal multi-GPU analogue of StochasticRowBlockDataset.

    - Shards edges by ROW across devices once
    - Each device keeps (S_index_shard, S_value_shard) resident on that GPU
    - reshuffle() is local within each shard (permutes only local rows)
    - __getitem__ returns per-device block info:
        (bid, edge_idx_local, row_indices_global, S_index_shard, S_value_shard)
      so your per-device compute can index into that shard without cross-GPU transfers.
    """

    # TODO: Typing/data class for shards
    shards: list[dict[str, Any]]
    S_index: torch.Tensor
    S_value: torch.Tensor
    m: int
    num_blocks: int
    device: list[torch.device]
    num_devices: int
    row_to_edges: list[torch.Tensor]
    gen: torch.Generator
    

    def __init__(self,
        S_index: torch.Tensor,
        S_value: torch.Tensor,
        m: int,
        num_blocks: int,
        gen: torch.Generator,
        devices: List[torch.device]
    ):
        self.m = int(m)
        self.num_blocks = int(num_blocks)
        self.devices = list(devices)
        self.num_devices = len(self.devices)
        self.gen = gen

        # Shard from CPU (recommended; avoids parking full S on one GPU)
        S_index_cpu = S_index.detach().cpu()
        S_value_cpu = S_value.detach().cpu()
        rows = S_index_cpu[0]

        # NEW: balanced span  #TODO: clean-up
        # spans = nnz_balanced_row_spans(rows, self.m, self.num_devices)

        self.shards = []
        for di, dev in enumerate(self.devices):
            row_start, row_end = block_span(di, self.m, self.num_devices) 
            # row_start, row_end = spans[di]
            mask = (rows >= row_start) & (rows < row_end)

            shard = {
                "device": dev,
                "row_start": int(row_start),
                "row_end": int(row_end),
                "m_local": int(row_end - row_start),
                "S_index": S_index_cpu[:, mask].to(dev, non_blocking=True),
                "S_value": S_value_cpu[mask].to(dev, non_blocking=True),
                "block_edge_idx": None,
                "block_row_indices": None,
            }
            self.shards.append(shard)

        # Ensure all asynchronous transfers to GPUs are complete before reshuffling
        for dev in self.devices:
            torch.cuda.synchronize(dev)
        self.reshuffle()


    # TODO: harmonize repetition with single-GPU version
    def reshuffle(self):
        """
        Local reshuffle per device: partitions ONLY local rows into num_blocks.
        """
        for shard in self.shards:
            dev = shard["device"]
            m_local = shard["m_local"]
            row_start = shard["row_start"]

            # 1) permute local rows [0, m_local)
            perm = torch.randperm(m_local, device=dev, generator=self.gen)
            perm_inv = torch.empty_like(perm)
            perm_inv[perm] = torch.arange(m_local, device=dev)

            # 2) map edges to blocks using local perm order
            block_size = (m_local + self.num_blocks - 1) // self.num_blocks

            edge_rows_global = shard["S_index"][0]           # in [row_start, row_end)
            edge_rows_local = edge_rows_global - row_start   # in [0, m_local)
            edge_block_ids = perm_inv[edge_rows_local] // block_size
            edge_block_ids.clamp_(max=self.num_blocks - 1)

            # 3) sort edges by block id + offsets
            sorted_edge_indices = torch.argsort(edge_block_ids)
            counts = torch.bincount(edge_block_ids, minlength=self.num_blocks)
            offsets = torch.zeros(self.num_blocks + 1, dtype=torch.long, device=dev)
            torch.cumsum(counts, dim=0, out=offsets[1:])

            # 4) materialize per-block edge idx + per-block (GLOBAL) row indices
            block_edge_idx = []
            block_row_indices = []
            for b in range(self.num_blocks):
                block_edge_idx.append(sorted_edge_indices[offsets[b]:offsets[b+1]])

                lp0 = b * block_size
                lp1 = min(m_local, (b + 1) * block_size)
                # perm[lp0:lp1] are local row ids; convert to global row ids
                block_row_indices.append(perm[lp0:lp1] + row_start)

            shard["block_edge_idx"] = block_edge_idx
            shard["block_row_indices"] = block_row_indices


    def __len__(self):
        return self.num_blocks


    def __getitem__(self, block_id: int):
        bid = int(block_id)
        # Return per-device block payloads
        return [
            (
                bid,
                shard["block_edge_idx"][bid],
                shard["block_row_indices"][bid],
                shard["S_index"],
                shard["S_value"]
            )
            for shard in self.shards
        ]


def get_sharded_ds(ds, devices, batch_block_ids):
    shards_per_device = [[] for _ in range(len(devices))]
    for di, shard in enumerate(ds.shards):
        # shard-local tensors (already on correct device)
        for bid in batch_block_ids:
            bid = int(bid)
            shards_per_device[di].append(
                (bid,
                    shard["block_edge_idx"][bid],
                    shard["block_row_indices"][bid])
            )
    return shards_per_device


### OLD: balanced sharded partition
def nnz_balanced_row_spans(rows_cpu: torch.Tensor, m: int, num_devices: int) -> List[Tuple[int, int]]:
    """
    Partition rows [0, m) into contiguous spans with approximately equal nnz per span.
    rows_cpu: S_index[0] on CPU (shape: nnz,)
    Returns list of (row_start, row_end) for each device, length=num_devices.
    """
    # nnz per row
    row_nnz = torch.bincount(rows_cpu, minlength=m)          # (m,)
    cum = torch.cumsum(row_nnz, dim=0)                       # (m,)
    total = int(cum[-1].item()) if m > 0 else 0

    # If matrix has no edges, fall back to equal row counts
    if total == 0:
        spans = []
        base = m // num_devices
        rem = m % num_devices
        start = 0
        for i in range(num_devices):
            end = start + base + (1 if i < rem else 0)
            spans.append((start, end))
            start = end
        return spans

    # target cumulative nnz boundaries
    # (avoid putting a cut at row 0 unless needed)
    targets = [total * (i + 1) // num_devices for i in range(num_devices - 1)]
    cuts = torch.searchsorted(cum, torch.tensor(targets, dtype=cum.dtype), right=False).tolist()

    # ensure non-decreasing and within [0, m]
    cuts = [max(0, min(m, int(c))) for c in cuts]
    # enforce monotonicity
    for i in range(1, len(cuts)):
        if cuts[i] < cuts[i - 1]:
            cuts[i] = cuts[i - 1]

    # build spans
    spans = []
    start = 0
    for c in cuts:
        spans.append((start, c))
        start = c
    spans.append((start, m))
    return spans
