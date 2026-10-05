# Process online

## Simulate it on a recording

`processing_mode="online"` sends every batch through the full real-time path: filtering,
channel selection, centring and extension per raw batch, with their state carried from one
batch to the next. Here the stream starts where the calibration window ends, as a live feed
would after calibrating:

```python
--8<-- "workflow.py:online"
```

This checks the per-batch pipeline on a known recording. The online and offline outputs
aren't bit-identical, since online centring uses a running mean rather than the whole
recording's.

## Process a live feed

With no recording to hand over, call `process_batch` in a loop you own, one batch of
`batch_size` raw samples (`batch_ms` of EMG) at a time, as the acquisition delivers them. Here
the recording, split into batches, stands in for the device:

```python
--8<-- "workflow.py:online-loop"
```

A fresh model treats each batch as raw EMG and carries the filter, centring and extension
state from one call to the next, so this loop gives the same output as
`processing_mode="online"`, batch for batch. Each call returns that batch's spikes and
sources, ready to drive a decoder or a display.
