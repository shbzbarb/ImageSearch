import sys
import os
import argparse
import time
try:
    import matplotlib.pyplot as plt
    from PIL import Image
    VISUALIZATION_ENABLED = True
except ImportError:
    print("Warning: Matplotlib and/or Pillow not found. Visualization will be disabled.")
    print("Install them with: pip install matplotlib Pillow")
    VISUALIZATION_ENABLED = False

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
    parser.add_argument("--visualize", action='store_true', help="Display the image with its classification result.")

    return parser.parse_args()

#Visualization Function
def visualize_classification_result(image_path: str, label: str, score: float):
    """
    Displays the input image with its predicted label and confidence score.

    Args:
        image_path (str): Path to the input image.
        label (str): The predicted label for the image.
        score (float): The confidence score for the prediction.
    """
    if not VISUALIZATION_ENABLED:
        print("Visualization disabled because required libraries (Matplotlib/Pillow) are not installed.")
        return

    try:
        img = Image.open(image_path)
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path} for visualization.")
        return
    except Exception as e:
        print(f"Error loading image {image_path} for visualization: {e}")
        return

    plt.figure(figsize=(6, 7))
    plt.imshow(img)
    title_text = f"Predicted: {label}\nScore: {score:.4f}"
    plt.title(title_text, fontsize=12)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

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
        predicted_label = classification_result['label']
        confidence_score = classification_result['score']
        print(f"  Predicted Label: {predicted_label}")
        print(f"  Confidence Score: {confidence_score:.4f}")

        if args.visualize:
            if VISUALIZATION_ENABLED:
                print("\nDisplaying visualization...")
                visualize_classification_result(args.input_image, predicted_label, confidence_score)
            else:
                print("\nVisualization requested but libraries are not available.")
    else:
        print("\nClassification failed.")
        if args.visualize:
            print("Visualization skipped as classification failed.")


    end_time = time.time()
    print("\n--- Classification Script Complete ---")
    print(f"Total time taken: {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()