# OPF Metric Aggregation

In `opf_task.py`, test metrics are first reduced inside each batch, then reduced across batches by Lightning with weight `batch.num_graphs`. In `opf_ac_dc_baseline.py`, the corresponding AC/DC metrics are reduced directly over the full test split with `numpy`/`pandas`.

For metrics defined on all buses, these two procedures are equivalent when each graph has the same number of buses. That assumption holds here because evaluation is performed per network and buses are not removed.

For metrics defined on a bus subset, such as PV or REF buses, exact equivalence requires the same number of relevant buses in every graph. This is not strictly guaranteed: for example, a PV bus can become PQ if its only generator is disconnected. In practice, shuffling makes the subset counts similar across batches, but the equality is only approximate.

For edge metrics, the same is observed: graphs can have different numbers of active branches because lines may be deactivated. Therefore batch-level graph weighting is not exactly the same as a direct mean over all edge rows in the dataset, but since the data is shuffled, then both implementation give reasonably similar results.
