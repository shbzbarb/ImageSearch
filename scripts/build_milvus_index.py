import sys
import os
import argparse
import numpy as np
import time

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.utils import load_config
from src.milvus_utils import (
    connect_milvus,
    create_collection,
    create_index,
    insert_data,
    disconnect_milvus
)

def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Load features and paths into a Milvus collection.")
    parser.add_argument("-f", "--features_path", required=True, help="Path to the .npy file containing feature vectors.")
    parser.add_argument("-p", "--paths_file", required=True, help="Path to the .txt file containing corresponding image paths.")
    return parser.parse_args()

def load_image_paths(filepath):
    """Loads image paths from a text file, one path per line."""
    print(f"Attempting to load image paths from: {filepath}")
    if not os.path.exists(filepath):
        print(f"Error: Paths file not found at {filepath}.")
        return None
    try:
        with open(filepath, 'r') as f:
            image_paths = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(image_paths)} image paths.")
        return image_paths
    except Exception as e:
        print(f"Error loading paths file '{filepath}': {e}")
        return None

def main():
    """Main function to build the Milvus index."""
    args = parse_arguments()

    print("--- Milvus Index Build Script ---")
    start_time = time.time()

    #1.LOADING CONFIGURATION
    print("Loading configuration...")
    config = load_config()
    if config is None:
        print("Error: Failed to load config.yaml. Exiting.")
        sys.exit(1)

    #Extracting required config values
    milvus_config = config.get('milvus', {})
    milvus_uri_relative = milvus_config.get('uri') #getting the relative URI from config
    collection_name = milvus_config.get('collection_name')
    metric_type = milvus_config.get('metric_type')
    index_type = milvus_config.get('index_type')
    index_params = milvus_config.get('index_params')

    model_config = config.get('model', {})
    embedding_dim = model_config.get('embedding_dim')

    #Path Resolution and Directory Creation
    if not milvus_uri_relative:
        print("Error: Missing 'uri' in milvus configuration of config.yaml.")
        sys.exit(1) #Exit if URI is missing

    #Constructing the absolute path for the Milvus URI relative to project root
    if milvus_uri_relative.startswith('./'):
        relative_path_part = milvus_uri_relative[2:] 
        milvus_uri_absolute = os.path.join(project_root, relative_path_part)
    elif not os.path.isabs(milvus_uri_relative):
         print(f"Warning: Milvus URI '{milvus_uri_relative}' is relative but doesn't start with './'. Resolving relative to project root '{project_root}'.")
         milvus_uri_absolute = os.path.join(project_root, milvus_uri_relative)
    else:
        milvus_uri_absolute = milvus_uri_relative

    #Ensure the parent directory for the database file exists
    milvus_db_dir = os.path.dirname(milvus_uri_absolute)
    try:
        if milvus_db_dir:
             print(f"Ensuring Milvus data directory exists: {milvus_db_dir}")
             os.makedirs(milvus_db_dir, exist_ok=True)
    except OSError as e:
         print(f"Error creating Milvus directory '{milvus_db_dir}': {e}")
         sys.exit(1)

    print(f"Using resolved absolute Milvus URI: {milvus_uri_absolute}")

    #Validating other required config values
    if not all([collection_name, metric_type, index_type, index_params, embedding_dim]):
        print("Error: Missing required configuration values in config.yaml (milvus or model section).")
        sys.exit(1)

    #2.LOADING FEATURES
    print(f"Loading features from: {args.features_path}")
    if not os.path.exists(args.features_path):
        print(f"Error: Features file not found at {args.features_path}. Exiting.")
        sys.exit(1)
    try:
        features = np.load(args.features_path)
        print(f"Loaded {features.shape[0]} feature vectors with dimension {features.shape[1]}.")
        #Validating feature dimensions match config
        if features.shape[1] != embedding_dim:
             print(f"Error: Feature dimension ({features.shape[1]}) in '{args.features_path}' "
                   f"does not match config embedding_dim ({embedding_dim}).")
             sys.exit(1)
    except Exception as e:
        print(f"Error loading features file '{args.features_path}': {e}")
        sys.exit(1)

    #3.LOADING IMAGE PATHS
    image_paths = load_image_paths(args.paths_file)
    if image_paths is None:
        print("Exiting due to error loading image paths.")
        sys.exit(1)

    #4.VALIDATING MATCHING LENGTHS
    if len(image_paths) != features.shape[0]:
        print(f"Error: Number of paths ({len(image_paths)}) in '{args.paths_file}' "
              f"does not match number of features ({features.shape[0]}) in '{args.features_path}'.")
        sys.exit(1)

    #5.CONNECTING TO MILVUS AND PERFORMING OPERATIONS
    milvus_connected = False
    collection = None
    try:
        #Attempting connection using the resolved absolute URI
        print("Attempting to connect to Milvus...")
        if not connect_milvus(milvus_uri_absolute):
             #If connect_milvus returns False, raise an error to be caught below
             raise ConnectionError(f"Failed to connect to Milvus using URI: {milvus_uri_absolute}")

        print("Milvus connection successful.")
        #Set flag only AFTER successful connection
        milvus_connected = True 

        #creating or geting collection
        print(f"Accessing/Creating collection: {collection_name}")
        collection = create_collection(collection_name, embedding_dim)
        if collection is None:
             raise RuntimeError("Failed to create or get Milvus collection.")

        #creating index on the collection
        print(f"Ensuring index exists on collection: {collection_name}")
        if not create_index(collection, metric_type, index_type, index_params):
             #Error messages printed within create_index
             raise RuntimeError("Failed to create or ensure Milvus index exists.")

        #insert data into the collection
        print(f"Starting data insertion into '{collection.name}'...")
        insert_result = insert_data(collection, image_paths, features)
        if insert_result is None:
             #Error message printed within insert_data
             raise RuntimeError("Error during data insertion.")

        print("Data insertion processing complete.")

        #IMPORTANT! Flush data to ensure it's written, indexed, and searchable
        print("Flushing collection to persist data...")
        collection.flush()
        print(f"Flush complete. Final entity count in '{collection.name}': {collection.num_entities}")

    except (ConnectionError, RuntimeError, Exception) as e:
        print(f"An error occurred during Milvus operations: {e}")
        sys.exit(1)

    finally:
        #6.DISCONNECT FROM MILVUS
        if milvus_connected:
             print("Disconnecting from Milvus...")
             disconnect_milvus()
        else:
             print("Milvus was not connected; skipping disconnect.")


    end_time = time.time()
    print("\n--- Milvus Index Build Complete ---")
    print(f"Successfully indexed {features.shape[0]} items.")
    print(f"Total time taken: {end_time - start_time:.2f} seconds.")


if __name__ == "__main__":
    main()