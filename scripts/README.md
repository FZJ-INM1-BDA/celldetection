# Scripts

## CPN Inference

### Minimal example
- Read and process inputs from `./inputs` directory (can also be a specific file name)
- Write outputs to `./outputs` directory
- Use models saved in `./models` directory (can also be a specific file name)
- Specified `tile_size` and `stride` are used for sliding window processing
  - Smaller `tile_size` consumes less memory, but usually more time
  - The `stride` parameter defines by how much the sliding window should move at each step
  - It is important to allow for a sufficiently large overlap between adjacent windows, as this avoids tiling artifacts (e.g. false negatives)

```
python scripts/cpn_inference.py -i ./inputs/ -o ./outputs/ -m ./models/ --tile_size 1024 --stride 728
```

### Include Region Properties
- Following the `-p` option, you may list the names of properties of interest
- A list of valid properties is available in the "Notes" subsection here: https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.regionprops
```
python scripts/cpn_inference.py -i ./inputs/ -o ./outputs/ -m ./models/ -p label area --tile_size 1024 --stride 728
```