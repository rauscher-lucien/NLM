import os
import sys
import argparse
import logging
import numpy as np
import tifffile
import cv2

class StreamToLogger(object):
    def __init__(self, logger, log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())

    def flush(self):
        pass

def setup_logging(log_file='logging.log'):
    logging.basicConfig(filename=log_file, filemode='a', format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console_handler)
    sys.stdout = StreamToLogger(logging.getLogger('STDOUT'), logging.INFO)
    sys.stderr = StreamToLogger(logging.getLogger('STDERR'), logging.ERROR)

def cuda_nlm_filter_3d_image_stack(image_stack, h, search_window, block_size):
    # Initialize CUDA device
    cuda = cv2.cuda

    # Initialize an empty array for the filtered stack, making sure to work with float32 for processing
    filtered_stack = np.zeros_like(image_stack, dtype=np.float32)

    # Apply CUDA NLM filtering slice by slice
    for i, slice in enumerate(image_stack):
        print(f"Processing slice {i}")
        # Convert to float32, assuming the original data range is 0-65535 for 16-bit
        slice_float = slice.astype(np.float32) / 65535.0
        
        # Upload the slice to GPU
        gpu_slice = cuda.GpuMat()
        gpu_slice.upload(slice_float)
        
        # Apply CUDA NLM filter to the slice
        filtered_gpu_slice = cuda.fastNlMeansDenoising(gpu_slice, h, search_window, block_size)
        
        # Download the filtered slice from GPU
        filtered_slice = filtered_gpu_slice.download()
        filtered_stack[i, :, :] = filtered_slice

    # Convert the result back to 16-bit if necessary, ensuring the data is properly scaled back to the 0-65535 range
    filtered_stack_16bit = np.clip(filtered_stack * 65535, 0, 65535).astype('uint16')
    return filtered_stack_16bit

def save_filtered_stack(filtered_stack, input_path):
    if filtered_stack is None:
        return
    
    # Generate the output filename based on the input filename
    base_name = os.path.basename(input_path)
    name, ext = os.path.splitext(base_name)
    output_filename = f"{name}_filtered_nlm_cuda{ext}"
    output_path = os.path.join(os.path.dirname(input_path), output_filename)

    # Save the filtered 3D stack
    tifffile.imwrite(output_path, filtered_stack)
    print(f"Filtered image stack saved at {output_path}")

def main():
    setup_logging()
    
    parser = argparse.ArgumentParser(description='Process inference parameters.')
    parser.add_argument('--data_path', type=str, help='Path to the data file', required=True)
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use for processing, e.g., "cuda:0" or "cpu"')
    parser.add_argument('--h', type=float, default=10, help='Filtering strength')
    parser.add_argument('--search_window', type=int, default=21, help='Size of the search window')
    parser.add_argument('--block_size', type=int, default=7, help='Size of the block to compare')

    args = parser.parse_args()

    data_path = args.data_path

    # Step 1: Read the 3D image stack
    image_stack = tifffile.imread(data_path)

    # Step 2: Apply CUDA NLM filtering to the image stack
    filtered_stack = cuda_nlm_filter_3d_image_stack(image_stack, args.h, args.search_window, args.block_size)

    # Step 3: Save the filtered 3D stack
    save_filtered_stack(filtered_stack, data_path)

if __name__ == '__main__':
    main()


