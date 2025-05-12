import sys
import os
import argparse
import numpy as np
from tqdm import tqdm

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.utils import load_config
from src.data_loader import find_image_files, load_and_preprocess_image
from src.feature_extractor import load_model_and_processor, extract_features

def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Extract features from images using a Vision Transformer.")
    parser.add_argument("-i", "--input_dir", required=True, help="Path to the directory containing input images")
    parser.add_argument("-o", "--output_prefix", required=True, help="Prefix for the output files. Features will be saved to prefix_features.npy and paths to prefix_paths.txt.")
    # parser.add_argument("-b", "--batch_size", type=int, default=None, help="Batch size for processing. Overrides config")
    
    return parser.parse_args()

def main():
    """Main function to run feature extraction."""
    args = parse_arguments()
    
    print("--- Feature Extraction Script ---")
    
    #1.LOADING CONFIG
    #uses default path "config.yaml"
    config = load_config()
    if config is None:
        print("Error: Failed to load config.yaml. Exiting.")
        sys.exit(1)
        
    output_dir = os.path.dirname(args.output_prefix)
    if output_dir and not os.path.exists(output_dir):
        print(f"Creating output directory: {output_dir}")
        os.makedirs(output_dir)
        
    output_features_path = f"{args.output_prefix}features.npy"
    output_paths_path = f"{args.output_prefix}paths.txt"

    #2.LOADING MODEL AND PROCESSOR
    print("Loading model and processor...")
    model, processor, device = load_model_and_processor()
    if model is None or processor is None:
        print("Error: Failed to load model or processor. Exiting.")
        sys.exit(1)
    print(f"Using device: {device}")

    #3.FINDING IMAGE FILES
    print(f"Finding images in: {args.input_dir}")
    image_paths = find_image_files(args.input_dir)
    if not image_paths:
        print("No image files found. Exiting.")
        sys.exit(0)

    #4.EXTRACTING FEATURES
    print(f"Starting feature extraction for {len(image_paths)} images...")
    feature_list = []
    valid_paths_list = []

    for image_path in tqdm(image_paths, desc="Extracting Features"): #tqdm for progress bar
        #loading and preprocessing single image
        image_tensor = load_and_preprocess_image(image_path, processor)
        
        if image_tensor is None:
            print(f"Warning: Skipping image {image_path} due to loading error.")
            continue

        #extracting features
        feature_vector = extract_features(image_tensor, model, device)

        if feature_vector is None:
            print(f"Warning: Skipping image {image_path} due to feature extraction error.")
            continue
            
        #storing successful results
        feature_list.append(feature_vector)
        valid_paths_list.append(image_path)

    if not feature_list:
        print("Error: No features were extracted successfully. Exiting.")
        sys.exit(1)
        
    print(f"\nSuccessfully extracted features for {len(feature_list)} images.")

    #5.SAVING RESULTS
    print(f"Saving features to: {output_features_path}")
    
    #converting list of numpy arrays to a single 2D numpy array
    features_np = np.array(feature_list)
    np.save(output_features_path, features_np)
    
    print(f"Saving corresponding image paths to: {output_paths_path}")
    with open(output_paths_path, 'w') as f:
        for path in valid_paths_list:
            f.write(f"{path}\n")

    print("--- Feature Extraction Complete ---")

if __name__ == "__main__":
    main()