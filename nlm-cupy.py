import cupy as cp
import numpy as np
import tifffile
import os

def nlm_filter_3d_image_stack(image_stack, h, search_window, block_size):
    # Convert the image stack to CuPy array
    image_stack = cp.asarray(image_stack, dtype=cp.float32) / 65535.0
    
    # Get the dimensions of the image stack
    depth, height, width = image_stack.shape
    
    # Initialize the output stack
    filtered_stack = cp.zeros_like(image_stack)
    
    # Define half window sizes
    half_search_window = search_window // 2
    half_block_size = block_size // 2
    
    # Precompute the Gaussian weights for the search window
    search_window_coords = cp.arange(-half_search_window, half_search_window + 1)
    gaussian_weights = cp.exp(-0.5 * (search_window_coords ** 2) / (half_search_window ** 2))
    gaussian_weights /= gaussian_weights.sum()
    
    for z in range(depth):
        print(f"Processing slice {z}")
        for y in range(height):
            for x in range(width):
                # Get the local patch around the current pixel
                y_min = max(0, y - half_search_window)
                y_max = min(height, y + half_search_window + 1)
                x_min = max(0, x - half_search_window)
                x_max = min(width, x + half_search_window + 1)
                
                # Extract the search window and the central block
                search_window = image_stack[z, y_min:y_max, x_min:x_max]
                central_block = image_stack[z, max(0, y - half_block_size):min(height, y + half_block_size + 1),
                                            max(0, x - half_block_size):min(width, x + half_block_size + 1)]
                
                # Compute the sum of squared differences
                ssd = cp.zeros_like(search_window)
                for dy in range(-half_block_size, half_block_size + 1):
                    for dx in range(-half_block_size, half_block_size + 1):
                        shifted = cp.roll(search_window, shift=(dy, dx), axis=(0, 1))
                        
                        # Calculate valid bounds for the shifted patch
                        y_start = max(0, dy + half_block_size)
                        y_end = min(search_window.shape[0], search_window.shape[0] + dy - half_block_size)
                        x_start = max(0, dx + half_block_size)
                        x_end = min(search_window.shape[1], search_window.shape[1] + dx - half_block_size)
                        
                        ssd[:(y_end-y_start), :(x_end-x_start)] += (
                            central_block - shifted[y_start:y_end, x_start:x_end]
                        ) ** 2
                
                # Compute the weights based on the SSD and Gaussian weights
                weights = cp.exp(-ssd / (h ** 2))
                weights *= gaussian_weights[half_search_window - (y - y_min):half_search_window + (y_max - y), 
                                            half_search_window - (x - x_min):half_search_window + (x_max - x)]
                weights /= weights.sum()
                
                # Compute the filtered pixel value
                filtered_stack[z, y, x] = (weights * search_window).sum()
    
    # Convert the result back to 16-bit
    filtered_stack = (filtered_stack * 65535).clip(0, 65535).astype(cp.uint16)
    
    return cp.asnumpy(filtered_stack)

def save_filtered_stack(filtered_stack, input_path):
    if filtered_stack is None:
        return
    
    # Generate the output filename based on the input filename
    base_name = os.path.basename(input_path)
    name, ext = os.path.splitext(base_name)
    output_filename = f"{name}_filtered_nlm_cupy{ext}"
    output_path = os.path.join(os.path.dirname(input_path), output_filename)

    # Save the filtered 3D stack
    tifffile.imwrite(output_path, filtered_stack)
    print(f"Filtered image stack saved at {output_path}")

def main():
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description='Process inference parameters.')
    parser.add_argument('--data_path', type=str, help='Path to the data file', required=True)
    parser.add_argument('--h', type=float, default=10, help='Filtering strength')
    parser.add_argument('--search_window', type=int, default=21, help='Size of the search window')
    parser.add_argument('--block_size', type=int, default=7, help='Size of the block to compare')

    args = parser.parse_args()

    data_path = args.data_path

    # Step 1: Read the 3D image stack
    image_stack = tifffile.imread(data_path)

    # Step 2: Apply NLM filtering to the image stack
    filtered_stack = nlm_filter_3d_image_stack(image_stack, args.h, args.search_window, args.block_size)

    # Step 3: Save the filtered 3D stack
    save_filtered_stack(filtered_stack, data_path)

if __name__ == '__main__':
    main()

