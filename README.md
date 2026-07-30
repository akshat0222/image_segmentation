# image_segmentation
An AI ppeline that isolates floors from room photos using ADE20K semantic segmentation to find SKUs via VGG Gram matrices and CIELAB color histograms.
# Floor Texture Similarity Ranking

## Project Overview

This project ranks the most similar SKU floor tile images for each query
image by segmenting the floor region, extracting texture and color
features, computing similarity scores, and ranking all SKU images.

## Approach and Pipeline

1.  Load SKU and query images.
2.  Segment floor regions using SegFormer-B0.
3.  Extract texture descriptors using ResNet-18 Layer1/Layer2 Gram
    matrices with rotation averaging.
4.  Extract color descriptors using normalized CIE Lab histograms.
5.  Compute:
    -   Texture Similarity: Cosine Similarity
    -   Color Similarity: Histogram Intersection
    -   Calculate Score = 0.45 × Texture + 0.55 × Color
6.  Rank SKU images and export results to CSV.

## Models and Techniques Used

-   **SegFormer-B0** (`nvidia/segformer-b0-finetuned-ade-512-512`) for
    floor segmentation. trained on ADE20K which can distinguish between object using its 150 classes like floor has 3 class id 
-   **ResNet-18** (ImageNet pretrained) for texture feature extraction.
-   **Gram Matrix** representation for texture.
-   **Rotation Averaging** for orientation robustness. needed as to get correct result for the images taken at different angle
-   **CIE Lab Color Space** and histogram intersection for color
    comparison.

## How Similarity is Computed

Texture similarity is calculated using cosine similarity between
normalized descriptors. Color similarity is calculated using histogram
intersection of normalized Lab histograms. The final similarity score
is: `0.45 × Texture Similarity + 0.55 × Color Similarity`.


