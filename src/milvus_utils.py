import os
from pymilvus import (
    connections, utility, CollectionSchema, FieldSchema, DataType, Collection,
    MilvusException
)
import numpy as np
from typing import List, Tuple, Dict, Any, Optional

from src.utils import load_config

CONFIG = load_config()
if CONFIG is None:
    raise RuntimeError("Failed to load configuration for Milvus Utils. Check config.yaml.")

#extracting Milvus configuration details 
MILVUS_CONFIG = CONFIG.get('milvus', {})

#DEFAULT_ALIAS
DEFAULT_ALIAS = 'default' 
EMBEDDING_FIELD_NAME = MILVUS_CONFIG.get('embedding_field_name', 'embedding') 
VECTOR_INDEX_NAME = MILVUS_CONFIG.get('index_name', 'idx_vector_embeddings') 

#Connection Management
#connect_milvus to accept uri
def connect_milvus(uri: str) -> bool:
    """
    Establishes a connection to Milvus using the provided URI.

    Args:
        uri (str): The connection URI for Milvus (e.g., an absolute path like 
                   '/path/to/project/milvus_data/milvus.db').

    Returns:
        True if connection is successful or already established, False otherwise.
    """
    try:
        if connections.has_connection(DEFAULT_ALIAS):
             print(f"Milvus connection ('{DEFAULT_ALIAS}') already exists.")
             return True
             
        print(f"Connecting to Milvus using URI: {uri} with alias '{DEFAULT_ALIAS}'...")
        connections.connect(alias=DEFAULT_ALIAS, uri=uri) 
        print("Successfully connected to Milvus.")
        return True
    except MilvusException as e:
        print(f"Error connecting to Milvus: {e}")
        return False
    except Exception as e:
        print(f"An unexpected error occurred during Milvus connection: {e}")
        return False

def disconnect_milvus():
    """Disconnects from Milvus."""
    try:
        if connections.has_connection(DEFAULT_ALIAS):
            print(f"Disconnecting from Milvus ('{DEFAULT_ALIAS}')...")
            connections.disconnect(DEFAULT_ALIAS)
            print("Successfully disconnected from Milvus.")
        else:
            print("No active Milvus connection to disconnect.")
    except Exception as e:
        print(f"Error disconnecting from Milvus: {e}")


#Collection and Index Management
def create_collection(collection_name: str, embedding_dim: int, 
                      description: str = "Image Embeddings Collection") -> Optional[Collection]:
    """
    Creates a Milvus collection with a predefined schema or returns 
    the existing one.
    (Schema definition remains the same)

    Args:
        collection_name: Name of the collection to create.
        embedding_dim: Dimension of the feature vectors to be stored.
        description: Optional description for the collection.

    Returns:
        A pymilvus Collection object, or None if creation fails or connection failed.
    """
    
    if not connections.has_connection(DEFAULT_ALIAS):
        print("Error: Milvus connection not established before calling create_collection.")
        return None

    try:
        if utility.has_collection(collection_name, using=DEFAULT_ALIAS):
            print(f"Collection '{collection_name}' already exists. Returning existing collection.")
            return Collection(name=collection_name, using=DEFAULT_ALIAS)

        print(f"Creating collection '{collection_name}' with embedding dimension {embedding_dim}...")
        
        pk_field = FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True)
        path_field = FieldSchema(name="image_path", dtype=DataType.VARCHAR, max_length=65535) 
        embedding_field = FieldSchema(name=EMBEDDING_FIELD_NAME, dtype=DataType.FLOAT_VECTOR, dim=embedding_dim) 
        
        schema = CollectionSchema(
            fields=[pk_field, path_field, embedding_field],
            description=description,
            enable_dynamic_field=False 
        )

        collection = Collection(
            name=collection_name,
            schema=schema,
            using=DEFAULT_ALIAS,
            consistency_level="Bounded" 
        )
        print(f"Collection '{collection_name}' created successfully.")
        return collection

    except MilvusException as e:
        print(f"Milvus error creating collection '{collection_name}': {e}")
        return None
    except Exception as e:
        print(f"Unexpected error creating collection '{collection_name}': {e}")
        return None


def create_index(collection: Collection, metric_type: str, index_type: str, 
                 index_params: Dict[str, Any]) -> bool:
    if not connections.has_connection(DEFAULT_ALIAS):
        print("Error: Milvus connection not established before calling create_index.")
        return False
    try:
        print(f"Ensuring index '{VECTOR_INDEX_NAME}' exists on field '{EMBEDDING_FIELD_NAME}' for collection '{collection.name}'...")
        
        index_full_params = {
            "metric_type": metric_type,
            "index_type": index_type,
            "params": index_params
        }

        collection.create_index(
            field_name=EMBEDDING_FIELD_NAME, 
            index_params=index_full_params,
            index_name=VECTOR_INDEX_NAME 
        )
        print(f"Index '{VECTOR_INDEX_NAME}' created successfully.")

    except MilvusException as e:
        error_message = str(e).lower()
        if "index already exist" in error_message or "duplicate index" in error_message:
             print(f"Index '{VECTOR_INDEX_NAME}' already exists.")
        elif "there are multiple indexes" in error_message:
             print(f"Warning: Milvus reports multiple indexes ('{e}'). Assuming desired vector index might exist or creation is blocked.")
        else:
            print(f"Milvus error creating index on '{collection.name}': {e}")
            return False 

    except Exception as e:
        print(f"Unexpected error during index creation check/attempt on '{collection.name}': {e}")
        return False

    try:
        print(f"Loading collection '{collection.name}'...")
        collection.load()
        print(f"Collection '{collection.name}' loaded successfully.")
        return True
    except Exception as e:
         print(f"Error loading collection '{collection.name}' after index check/creation: {e}")
         return False

def insert_data(collection: Collection, paths: List[str], 
                embeddings: np.ndarray) -> Optional[List[Any]]:
    if not connections.has_connection(DEFAULT_ALIAS):
        print("Error: Milvus connection not established before calling insert_data.")
        return None
        
    if len(paths) != embeddings.shape[0]:
        print("Error: Number of paths does not match number of embeddings.")
        return None
        
    data_to_insert = [
        paths,
        embeddings.tolist() 
    ]

    try:
        print(f"Inserting {len(paths)} entities into collection '{collection.name}'...")
        mutation_result = collection.insert(data_to_insert)
        print(f"Insertion successful. Primary keys (first 10): {mutation_result.primary_keys[:10]}...") 
        return mutation_result 

    except MilvusException as e:
        print(f"Milvus error inserting data into '{collection.name}': {e}")
        return None
    except Exception as e:
        print(f"Unexpected error inserting data into '{collection.name}': {e}")
        return None

def search_vectors(collection: Collection, query_vectors: np.ndarray, top_k: int, 
                   search_params: Dict[str, Any]) -> Optional[List[List[Dict[str, Any]]]]:
    if not connections.has_connection(DEFAULT_ALIAS):
        print("Error: Milvus connection not established before calling search_vectors.")
        return None
        
    try:
        print(f"Searching collection '{collection.name}' with {query_vectors.shape[0]} queries...")
        metric_type = 'L2' 
        vec_index_found = False
        if collection.has_index:
             for index in collection.indexes:
                 if index.index_name == VECTOR_INDEX_NAME or index.field_name == EMBEDDING_FIELD_NAME:
                     metric_type = index.params.get('metric_type', metric_type) 
                     vec_index_found = True
                     print(f"  Using Metric Type from index '{index.index_name}': {metric_type}")
                     break
        if not vec_index_found:
             print(f"Warning: Could not find specific vector index '{VECTOR_INDEX_NAME}' or index on field '{EMBEDDING_FIELD_NAME}'. Using default metric type '{metric_type}' for search.")
             
        print(f"  Top K: {top_k}")
        print(f"  Search Params: {search_params}")
        search_full_params = {"metric_type": metric_type, "params": search_params}

        search_results = collection.search(
            data=query_vectors,
            anns_field=EMBEDDING_FIELD_NAME, 
            param=search_full_params, 
            limit=top_k,
            output_fields=["image_path"] 
        )
        print(f"Search completed. Found results for {len(search_results)} queries.")
        
        processed_results = []
        for hits in search_results:
            query_hits = []
            for hit in hits:
                hit_data = {
                    "id": hit.id,
                    "distance": hit.distance,
                    "image_path": hit.entity.get('image_path', None) 
                }
                query_hits.append(hit_data)
            processed_results.append(query_hits)
        return processed_results

    except MilvusException as e:
        print(f"Milvus error searching collection '{collection.name}': {e}")
        return None
    except Exception as e:
        print(f"Unexpected error searching collection '{collection.name}': {e}")
        return None