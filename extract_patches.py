from histoprep import SlideReader
import os
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

# Paths for OCSCC dataset
slides_path = "/path/to/OCSCC/slides"  # Directory containing OCSCC WSIs
masks_path = "/path/to/OCSCC/epithelial_masks"  # Directory containing epithelial annotations
save_path = "home/OCSCC/patches"  # Where to save extracted patches
mask_save_path = "home/OCSCC/masks"  # Where to save corresponding mask patches

# Parameters adjusted for OCSCC epithelial segmentation
level = 1  # Magnification level to extract patches
overlap = 0  # No overlap between patches
max_background = 0.5  # Maximum allowed background in a patch
patch_size = 2000  # Larger patches for OCSCC

def extract_patches_from_wsi(wsi_path, mask_path, wsi_id):
    """
    Extract patches from a single WSI and its corresponding epithelial mask
    """
    reader = SlideReader(wsi_path)
    downsample = reader.level_downsamples[level][1]
    downsampled_patch_size = int(patch_size * downsample)
    
    # Read mask using PIL backend
    mask_reader = SlideReader(mask_path, backend="PILLOW")
    
    # Get tissue mask and threshold
    threshold, tissue_mask = reader.get_tissue_mask(level=level)
    
    # Get coordinates for tiles
    tile_coordinates = reader.get_tile_coordinates(
        tissue_mask,
        width=downsampled_patch_size,
        overlap=overlap * downsample,
        max_background=max_background,
    )
    
    # Save patches and masks
    os.makedirs(os.path.join(save_path, wsi_id), exist_ok=True)
    os.makedirs(os.path.join(mask_save_path, wsi_id), exist_ok=True)
    
    tile_metadata = reader.save_regions(
        os.path.join(save_path, wsi_id),
        tile_coordinates,
        level=level,
        threshold=threshold,
        save_metrics=False,
        overwrite=True,
    )
    
    mask_metadata = mask_reader.save_regions(
        os.path.join(mask_save_path, wsi_id),
        tile_coordinates,
        level=level,
        threshold=threshold,
        save_metrics=False,
        overwrite=True,
    )
    
    return len(tile_coordinates)

def main():
    """
    Process all WSIs in the OCSCC dataset
    """
    # Get list of WSIs
    wsis = [f for f in os.listdir(slides_path) if f.endswith('.svs')]
    
    total_patches = 0
    for wsi in wsis:
        wsi_id = wsi.split('.')[0]
        wsi_path = os.path.join(slides_path, wsi)
        mask_path = os.path.join(masks_path, f"{wsi_id}.png")
        
        if os.path.exists(mask_path):
            print(f"Processing WSI: {wsi_id}")
            n_patches = extract_patches_from_wsi(wsi_path, mask_path, wsi_id)
            total_patches += n_patches
            print(f"Generated {n_patches} patches for {wsi_id}")
        else:
            print(f"Warning: No mask found for {wsi_id}")
    
    print(f"Total patches extracted: {total_patches}")

if __name__ == "__main__":
    main()