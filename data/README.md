# Dataset packaging

This directory separates redistributable benchmark assets from third-party data.

## Included

- `processed/synthetic_demo.npz`: the repository's generated synthetic demo.
- `manifests/safe_twitter.json`: audited statistics, time semantics and source
  checksums for the locally adapted SAFE-Twitter split.
- `manifests/safe_wiki.json`: the equivalent audit record for SAFE-Wiki.

## Not included

The SAFE-Twitter and SAFE-Wiki `.npy`/`.npz` arrays are not redistributed in
this branch because the upstream repository does not state a redistribution
license. Obtain the data from the upstream SAFE repository and run:

```bash
python scripts/adapt_safe_legacy_data.py \
  --dataset twitter \
  --source data/third_party/SAFE/twitter \
  --output data/adapted/twitter

python scripts/adapt_safe_legacy_data.py \
  --dataset wiki \
  --source data/third_party/SAFE/wiki/v8 \
  --output data/adapted/wiki
```

Upstream data source: <https://github.com/PanpanZheng/SAFE>

The generated manifest deliberately records a logical source label rather than
an absolute local path. For both datasets, `t_onset=-1`: the files contain an
observed suspension/block endpoint, not an independently verified fraud onset.
