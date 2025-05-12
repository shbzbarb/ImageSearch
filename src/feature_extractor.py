import torch
from transformers import AutoProcessor, AutoModel
from typing import Tuple, Any, Optional
import numpy as np

from src.utils import load_config 

CONFIG = load_config() 
if CONFIG is None:
    raise RuntimeError("Failed to load configuration. Check config.yaml and src/utils.py")

MODEL_NAME = CONFIG.get('model', {}).get('name', None)
if MODEL_NAME is None:
    raise ValueError("Model name not found in config.yaml under model.name")

def load_model_and_processor() -> Tuple[Optional[Any], Optional[Any], Optional[str]]:
    """
    Loads the pre-trained Vision Transformer model and its associated processor
    specified in the config file (model.name). 
    Determines the best available device (CUDA or CPU).

    Returns:
        A tuple containing:
        - The loaded model (moved to the best device).
        - The loaded processor.
        - The name of the device being used ('cuda:0' or 'cpu').
        Returns (None, None, None) if loading fails.
    """
    try:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"Loading model '{MODEL_NAME}' onto device '{device}'...")

        #loading the processor associated with the model
        processor = AutoProcessor.from_pretrained(MODEL_NAME)

        #loading the pre-trained model
        model = AutoModel.from_pretrained(MODEL_NAME)

        #moving the model to the selected device
        model.to(device)

        #setting the model to evaluation mode
        model.eval() 

        print(f"Model '{MODEL_NAME}' and processor loaded successfully.")
        return model, processor, device

    except Exception as e:
        print(f"Error loading model or processor '{MODEL_NAME}': {e}")
        return None, None, None

#Extract Features Function
def extract_features(image_tensor: torch.Tensor, model: Any, device: str) -> Optional[np.ndarray]:
    """
    Extracts features (embeddings) from a preprocessed image tensor using 
    the loaded model.
    (Function body is unchanged - keeping it short here)
    """
    if image_tensor is None or model is None:
        return None
    try:
        image_tensor = image_tensor.to(device)
        with torch.no_grad():
            outputs = model(pixel_values=image_tensor)
        
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            features = outputs.pooler_output 
        elif hasattr(outputs, 'last_hidden_state') and outputs.last_hidden_state is not None:
            features = outputs.last_hidden_state[:, 0, :] 
        else:
            print("Warning: Could not find standard 'pooler_output' or 'last_hidden_state'.")
            print(f"Output keys: {outputs.keys()}") 
            return None
            
        features_np = features.squeeze().cpu().numpy() 
        return features_np
    except Exception as e:
        print(f"Error during feature extraction: {e}")
        return None