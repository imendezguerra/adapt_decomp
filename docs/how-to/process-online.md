# Process online

`processing_mode="online"` sends every batch through the full real-time path: filtering,
channel selection, centring and extension per raw batch, with their state carried from one
batch to the next.

```python
--8<-- "workflow.py:online"
```

This checks the per-batch pipeline on a known recording. For a live feed, with no recording
to hand over, call `process_batch` in a loop you own; see
[Online simulation and full online mode](../adaptation.md#online-simulation-and-full-online-mode).
The online and offline outputs aren't bit-identical, since online centring uses a running
mean rather than the whole recording's.
