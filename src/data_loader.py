import os
from typing import List, Any, Optional
from PIL import Image
import torch

SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}

def find_image_files(data_dir: str) -> List[str]:
    """
    Recursively finds all image files in a directory

    Args:
        data_dir: The root directory to search within

    Returns:
        A list of full paths to the found image files
    """
    image_files = []
    if not os.path.isdir(data_dir):
        print(f"Error: Directory not found - {data_dir}")
        return image_files

    for root, _, files in os.walk(data_dir):
        for file in files:
            if os.path.splitext(file.lower())[1] in SUPPORTED_EXTENSIONS:
                image_files.append(os.path.join(root, file))

    print(f"Found {len(image_files)} image files in {data_dir}")
    return image_files

def load_and_preprocess_image(image_path: str, processor: Any) -> torch.Tensor | None:
    """
    Loads an image, preprocesses it using the provided Hugging Face processor,
    and returns the tensor ready for the model

    Args:
        image_path: Path to the image file
        processor: An instance of a Hugging Face image processor

    Returns:
        A PyTorch tensor representing the processed image (usually includes
        resizing, normalization, and conversion to tensor), or None if
        the image cannot be opened. The tensor shape is typically
        [batch_size, num_channels, height, width].
    """
    try:
        #loading the image using Pillow
        img = Image.open(image_path).convert("RGB")
        
        #processing the image, processor handles resizing, normalization, to_tensor
        inputs = processor(images=img, return_tensors="pt")

        #return the pixel values tensor
        #shape is expected to be [1, C, H, W] by default
        return inputs['pixel_values']

    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return None
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None