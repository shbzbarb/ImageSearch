import sys
import os
import argparse
import time

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.utils import load_config 
from src.data_loader import load_and_preprocess_image 
from src.classification import (
    load_classification_model, 
    classify_image, 
    DEFAULT_CLASSIFICATION_MODEL 
)

def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Classify a single image.")
    parser.add_argument("-i", "--input_image", required=True, help="Path to the input image file.")
    # parser.add_argument("-m", "--model_name", default=None, help="Name of classification model to use. Overrides defaults.")
    
    return parser.parse_args()

def main():
    """Main function to run classification."""
    args = parse_arguments()
    
    print("--- Image Classification Script ---")
    start_time = time.time()

    #1.LOAD CONFIGURATION
    model_name = DEFAULT_CLASSIFICATION_MODEL 

    if not os.path.exists(args.input_image):
        print(f"Error: Input image not found at {args.input_image}")
        sys.exit(1)

    #2.LOADING MODEL AND PROCESSOR
    print(f"Loading classification model '{model_name}'...")
    model, processor, device = load_classification_model(model_name) 
    if model is None or processor is None:
        print("Error: Failed to load model or processor. Exiting.")
        sys.exit(1)
    print(f"Using device: {device}")

    #3.LOADING AND PROCESSING INPUT IMAGE
    print(f"Loading and processing image: {args.input_image}")
    image_tensor = load_and_preprocess_image(args.input_image, processor) 
    if image_tensor is None:
        print("Error: Failed to load or process input image. Exiting.")
        sys.exit(1)

    #4.CLASSIFYING IMAGES
    print("Classifying image...")
    classification_result = classify_image(image_tensor, model, device)
    
    #5.SHOWING RESULTS
    if classification_result:
        print("\n--- Classification Result ---")
        print(f"  Predicted Label: {classification_result['label']}")
        print(f"  Confidence Score: {classification_result['score']:.4f}")
    else:
        print("\nClassification failed.")

    end_time = time.time()
    print("\n--- Classification Script Complete ---")
    print(f"Total time taken: {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()