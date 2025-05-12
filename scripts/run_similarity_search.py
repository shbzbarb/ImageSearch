import sys
import os
import argparse
import numpy as np
import time
import math

try:
    import matplotlib.pyplot as plt
    from PIL import Image
    VISUALIZATION_ENABLED = True
except ImportError:
    print("Warning: Matplotlib or Pillow not found. Visualization disabled.")
    print("Install them with: pip install matplotlib Pillow")
    VISUALIZATION_ENABLED = False

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.utils import load_config
from src.data_loader import load_and_preprocess_image
from src.feature_extractor import load_model_and_processor, extract_features
from src.milvus_utils import (
    connect_milvus, 
    search_vectors,
    disconnect_milvus,
    Collection
)

def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Search for similar images in Milvus.")
    parser.add_argument("-q", "--query_image", required=True, help="Path to the query image file (e.g., from data/raw/dataset1).")
    parser.add_argument("-k", "--top_k", type=int, default=None, help="Number of similar results to retrieve (overrides config if set).")
    parser.add_argument("-visualize", "--visualize", action='store_true', help="Display a plot visualizing the query and result images.")

    return parser.parse_args()

#Visualization Function
def visualize_search_results(query_image_path, search_hits, results_title="Similarity Search Results"):
    if not VISUALIZATION_ENABLED:
        print("Visualization libraries not available. Skipping plot.")
        return

    num_results = len(search_hits)
    total_images = 1 + num_results
    if total_images <= 1: rows, cols = 1, 1
    elif total_images <= 4: rows, cols = 1, total_images
    else:
        cols = math.ceil(math.sqrt(total_images))
        rows = math.ceil(total_images / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    fig.suptitle(results_title, fontsize=16)
    if rows == 1 and cols == 1: axes = np.array([axes])
    axes = axes.flatten()

    try: 
        #Display Query Image
        query_img = Image.open(query_image_path)
        axes[0].imshow(query_img)
        axes[0].set_title(f"Query Image\n({os.path.basename(query_image_path)})")
        axes[0].axis('off')
    except Exception as e:
        print(f"Error loading query image {query_image_path}: {e}")
        axes[0].set_title("Query Image\n(Load Error)")
        axes[0].axis('off')

    if num_results > 0: 
        #Display Result Images
        for i, hit in enumerate(search_hits):
            ax_idx = i + 1
            if ax_idx >= len(axes): break
            try:
                res_img_path = hit['image_path']
                distance = hit['distance']
                img = Image.open(res_img_path)
                axes[ax_idx].imshow(img)
                axes[ax_idx].set_title(f"Rank {i+1}\nDist: {distance:.4f}\n({os.path.basename(res_img_path)})")
                axes[ax_idx].axis('off')
            except Exception as e:
                print(f"Error loading/displaying result image {hit.get('image_path', 'N/A')}: {e}")
                axes[ax_idx].set_title(f"Rank {i+1}\n(Load Error)")
                axes[ax_idx].axis('off')
    elif total_images > 1:
        axes[1].set_title("No Results Found")
        axes[1].axis('off')

    #hiding unused axes
    for i in range(total_images, len(axes)): axes[i].axis('off')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def main():
    """Main function to run similarity search."""
    args = parse_arguments()

    print("--- Image Similarity Search Script ---")
    start_time = time.time()

    #1.LOADING CONFIG
    print("Loading configuration...")
    config = load_config()
    if config is None:
        print("Error: Failed to load config.yaml. Exiting.")
        sys.exit(1)

    #2.EXTRACT REQUIRED CONFIG VALUES
    milvus_config = config.get('milvus', {})
    milvus_uri_relative = milvus_config.get('uri') # Get relative URI
    collection_name = milvus_config.get('collection_name')
    search_params = milvus_config.get('search_params')

    search_config = config.get('search', {})
    top_k = args.top_k if args.top_k is not None else search_config.get('top_k', 10)

    #Path Resolution and Directory Check
    if not milvus_uri_relative:
        print("Error: Missing 'uri' in milvus configuration of config.yaml.")
        sys.exit(1)

    #Construct the absolute path for the Milvus URI relative to project root
    if milvus_uri_relative.startswith('./'):
        relative_path_part = milvus_uri_relative[2:]
        milvus_uri_absolute = os.path.join(project_root, relative_path_part)
    elif not os.path.isabs(milvus_uri_relative):
        print(f"Warning: Milvus URI '{milvus_uri_relative}' is relative but doesn't start with './'. Resolving relative to project root '{project_root}'.")
        milvus_uri_absolute = os.path.join(project_root, milvus_uri_relative)
    else:
        #Assume absolute path
        milvus_uri_absolute = milvus_uri_relative

    milvus_db_dir = os.path.dirname(milvus_uri_absolute)
    try:
        if milvus_db_dir:
             os.makedirs(milvus_db_dir, exist_ok=True)
    except OSError as e:
         print(f"Error accessing/creating Milvus directory '{milvus_db_dir}': {e}")
         sys.exit(1)

    print(f"Using resolved absolute Milvus URI: {milvus_uri_absolute}")

    if not all([collection_name, search_params]):
        print("Error: Missing required configuration in config.yaml (milvus section: collection_name, search_params).")
        sys.exit(1)

    if not os.path.exists(args.query_image):
        print(f"Error: Query image not found at {args.query_image}")
        sys.exit(1)

    #2.LOADING MODEL AND PROCESSOR
    print("Loading model and processor...")
    model, processor, device = load_model_and_processor()
    if model is None or processor is None: sys.exit(1)
    print(f"Using device: {device}")

    #3.PROCESSING QUERY IMAGE
    print(f"Loading and processing query image: {args.query_image}")
    query_tensor = load_and_preprocess_image(args.query_image, processor)
    if query_tensor is None: sys.exit(1)

    #4.EXTRACT QUERY FEATURES
    print("Extracting features for query image...")
    query_vector = extract_features(query_tensor, model, device)
    if query_vector is None: sys.exit(1)
    query_vector_reshaped = query_vector.reshape(1, -1)
    print(f"Query vector extracted (shape: {query_vector_reshaped.shape}).")

    #5.CONNECTING TO MILVUS & SEARCH
    milvus_connected = False
    search_results = None
    collection = None
    try:
        #Passing absolute URI to connect_milvus
        if not connect_milvus(milvus_uri_absolute): #passing the resolved URI
            raise ConnectionError(f"Failed to connect to Milvus using URI: {milvus_uri_absolute}")
        milvus_connected = True

        print(f"Accessing collection: {collection_name}")
        collection = Collection(name=collection_name)

        print("Loading collection into memory...")
        collection.load() #loading data into memory for search
        print("Collection loaded.")

        # Perform the search
        search_results = search_vectors(collection, query_vector_reshaped, top_k, search_params)

    except (ConnectionError, RuntimeError, Exception) as e:
        print(f"An error occurred during Milvus operations: {e}")

    finally:
        #6.DISCONNECT FROM MILVUS
        if milvus_connected:
             disconnect_milvus()

    #7.DISPLAYING RESULTS
    hits = []
    if search_results is not None:
        print("\n--- Search Results ---")
        hits = search_results[0]
        if not hits:
            print("No similar images found.")
        else:
            c_name = collection.name if collection else collection_name
            print(f"Top {len(hits)} similar images found in '{c_name}':")
            for i, hit in enumerate(hits):
                 img_path = hit.get('image_path', 'N/A')
                 distance = hit.get('distance', float('nan'))
                 print(f"  {i+1}. Path: {img_path}, Distance: {distance:.4f}")

    else:
         print("\nSearch failed.")

    #8.VISUALIZATION
    if args.visualize:
        if search_results is not None:
             if hits:
                 print("\nGenerating visualization...")
                 visualize_search_results(args.query_image, hits)
             else:
                 print("\nGenerating visualization for query image only (no results found)...")
                 visualize_search_results(args.query_image, [])
        else:
             print("\nVisualization skipped as search failed.")


    end_time = time.time()
    print("\n--- Search Script Complete ---")
    print(f"Total time taken: {end_time - start_time:.2f} seconds.")


if __name__ == "__main__":
    main()