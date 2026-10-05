# Record provenance

`build_metadata` gathers:

- when a result was produced and how long it took;
- the machine: OS, CPU, GPU and memory;
- the Python and package versions;
- the git remote, commit and uncommitted changes;
- the lines that reproduce it.

`write_metadata` stores it as YAML next to the result.

```python
--8<-- "workflow.py:provenance"
```

The `reproduce` entry starts with the lines that rebuild the exact source tree, in the style
of wandb:

```text
git clone https://github.com/imendezguerra/adapt_decomp.git
cd adapt_decomp
git checkout -b "docs-example" <commit>
git apply data/fdsi_benchmark/outputs/docs-example/patches/<hash>.patch   # only with uncommitted changes
python docs/snippets/workflow.py
```

Uncommitted changes are saved once per distinct diff under `patch_dir`. Untracked files are
listed but not saved, so commit them for results that matter. The
[FDSI benchmark](../benchmarks/fdsi.md) writes such a file next to every output.
