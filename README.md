## Corrosion Detection
A small project about corrosion detection (course project).
The pipeline is as follows:

For a given image, use `resnet` for image classification. If it's classified as rust, use `unet` to get the corrosion mask.

### Notes
All the image from classification dataset are collected from internet (some of them are from pet datasets). The small dataset for corrosion segmentation is annotated by myself. Because of the lack of large dataset, the performance might be limited.
### Selected results
