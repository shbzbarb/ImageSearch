import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
from typing import Tuple, Any, Optional, Dict

from src.utils import load_config 

DEFAULT_CLASSIFICATION_MODEL = 'google/vit-base-patch16-224'


def load_classification_model(
    model_name: str = DEFAULT_CLASSIFICATION_MODEL
) -> Tuple[Optional[Any], Optional[Any], Optional[str]]:
    """
    Loads a pre-trained Image Classification model and its associated processor
    from Hugging Face. Determines the best available device (CUDA or CPU).

    Args:
        model_name: The name of the classification model on Hugging Face Hub 

    Returns:
        A tuple containing:
        - The loaded model (moved to the best device).
        - The loaded processor.
        - The name of the device being used ('cuda:0' or 'cpu').
        Returns (None, None, None) if loading fails.
    """
    try:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"Loading classification model '{model_name}' onto device '{device}'...")

        #loading the processor associated with the model
        #using AutoImageProcessor for classification models
        processor = AutoImageProcessor.from_pretrained(model_name)

        #loading the pre-trained model with the classification head
        model = AutoModelForImageClassification.from_pretrained(model_name)

        #moving the model to the selected device
        model.to(device)

        #setting the model to evaluation mode
        model.eval() 

        print(f"Classification model '{model_name}' and processor loaded successfully.")
        return model, processor, device

    except Exception as e:
        print(f"Error loading classification model or processor '{model_name}': {e}")
        return None, None, None

def classify_image(
    image_tensor: torch.Tensor, 
    model: Any, 
    device: str
) -> Optional[Dict[str, Any]]:
    """
    Classifies an image using the provided model.

    Args:
        image_tensor: The preprocessed image tensor (output from 
                      load_and_preprocess_image), expected shape [1, C, H, W].
        model: The loaded Hugging Face classification model (on correct device, eval mode).
        device: The device the model and tensor should be on ('cuda:0' or 'cpu').

    Returns:
        A dictionary containing 'label' (predicted class name) and 
        'score' (probability), or None if classification fails.
    """
    if image_tensor is None or model is None:
        return None
        
    try:
        #confirming that the input tensor is on the same device as the model
        image_tensor = image_tensor.to(device)

        #performing inference without calculating gradients
        with torch.no_grad():
            outputs = model(pixel_values=image_tensor)
            logits = outputs.logits

        #getting probabilities using softmax and find the prediction
        probabilities = torch.softmax(logits, dim=-1)
        predicted_index = torch.argmax(probabilities, dim=-1).item()
        
        #getting the score of the predicted class
        predicted_score = probabilities[0, predicted_index].item()

        #mapping index to label name using model's config
        predicted_label = model.config.id2label[predicted_index]

        return {"label": predicted_label, "score": predicted_score}

    except Exception as e:
        print(f"Error during image classification: {e}")
        return None