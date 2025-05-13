# Image Analysis and Similarity Search Project
This project provides a suite of tools for advanced image analysis, including robust feature extraction, accurate image classification, and efficient similarity search. The project leverages state-of-the-art deep learning models and a high-performance vector database to deliver powerful image understanding capabilities.

## Example Outputs
### Image Classification
<table>
  <tr>
    <td align="center">
      <img src="https://github.com/shbzbarb/ImageSearch/blob/main/classification1.png?raw=true" alt="Classification Example 1" width="350"/>
      <br/> </td>
    <td align="center">
       <img src="https://github.com/shbzbarb/ImageSearch/blob/main/classification2.png?raw=true" alt="Classification Example 2" width="350"/>
      <br/>
      </td>
  </tr>
</table>

### Image Similarity Search
<img src="https://github.com/shbzbarb/ImageSearch/blob/main/imageSimilarity.png?raw=true" alt="Image Similarity Search Example" width="600"/>


## Overview
The core idea behind this project is to utilize and integrate existing open-source resources for common and advanced image analysis tasks. The project specifically focuses on image feature extraction, image classification, and image similarity search. The project leverages deep learning models to understand image content and a vector database (Milvus Lite) for efficient storage and retrieval of similar images based on their semantic features.


## Key Features
- **Image Classification**: Predicts and assigns a descriptive label to an input image using pre-trained vision transformer models (Google's ViT)

- **Image Feature Extraction**: Converts images into dense numerical vectors (embeddings). These vectors capture the semantic essence of the images, enabling quantitative comparison and understanding

- **Image Similarity Search**: Finds images within a collection that are visually and semantically similar to a given query image. This is achieved by: 
    * **1.** Extracting the feature vector of the query image
    * **2.** Comparing this vector against a database of pre-computed feature vectors stored in Milvus Lite
    * **3.** Retrieving and displaying the images corresponding to the closest vectors in the feature space

- **Configurable Backend**: Utilizes a ```config.yaml``` for easy modification of model choices, paths, and database parameters

- **Script-based Workflow**: Provides individual Python scripts for each core functionality, allowing for modular execution and integration into larger pipelines

- **Visualization**: The project offers options to visualize search and classification results


## Technology Stack
The project utilizes the following core libraries and technologies:

* **Deep Learning Framework:**
    * **PyTorch:** Used for loading pre-trained models from Hugging Face and performing inference
    * **CUDA Support:** Enabled for GPU acceleration to speed up model inference

* **Model Hub & Utilities:**
    * **HuggingFace Transformers:** Provides access to state-of-the-art pre-trained models for image classification and feature extraction tasks

* **Vector Database:**
    * **Milvus Lite (`pymilvus`):**  An embedded, lightweight vector database used for efficient storage and fast similarity searching of the extracted image feature vectors

* **Image Processing:**
    * **Pillow(PIL):** Used for loading and basic image manipulation

* **Data Handling & Numerics:**
    * **NumPy**: For efficient numerical operations, especially with feature vectors

* **Visualization:**
    * **Matplotlib**: Used to display images and their associated metadata (for example classification labels or search ranks)

* **Development:**
    * **conda**: For virtual environment management
    * **pip**: For package installation

* **Example Dataset**: The concepts and scripts can be applied to various image datasets. The LIVE Image Quality Assessment Database [Link](https://live.ece.utexas.edu/research/Quality/subjective.htm) is one example of a publicly available image set.


## Architectural Overview
The project follows a modular architecture, with distinct components for different stages of the image analysis pipeline:

1.  Configuration (`config.yaml`): 
  * The YAML file defines parameters such as model names, embedding dimensions, Milvus connection details, file paths, and processing settings
  * This allows users to customize the behavior without modifying the core scripts.

2.  Data Loading (`src/data_loader.py`):
	* Responsible for finding image files in specified directories
	* Loads and preprocesses individual images using the appropriate processor from the Hugging Face Transformers library, making them ready for model input

3.  Feature Extraction (`src/feature_extractor.py`, `scripts/run_feature_extraction.py`):
	* Loads a pre-trained vision model and its associated processor
	* The `run_feature_extraction.py` script iterates through images from an input directory, uses `feature_extractor.py` to convert each image into a high-dimensional feature vector (embedding), and saves these vectors along with their corresponding image paths

4.  Milvus Indexing (`src/milvus_utils.py`, `scripts/build_milvus_index.py`):
	* `milvus_utils.py` handles all interactions with the Milvus Lite database, including connecting, creating collections, defining schemas, inserting data, creating vector indexes, and searching
	* The `build_milvus_index.py` script takes the features and paths generated by the feature extraction step and populates a Milvus collection. It creates an efficient index on the feature vectors to enable fast similarity searches

5.  Image Classification (`src/classification.py`, `scripts/run_classification.py`):
	* `classification.py` loads a pre-trained image classification model and its processor. It provides a function to predict the class label and confidence for a given image tensor
	* `run_classification.py` is a command-line tool to classify a single image and display the top predicted label and score, with an option to visualize the image and result

6.  Similarity Search (`scripts/run_similarity_search.py`):
	* This script takes a query image as input
	* It extracts the feature vector for the query image using the same model and processor used for indexing
	* It then queries the Milvus database (using functions from `milvus_utils.py`) to find the most similar image vectors from the indexed collection
	* Finally, it displays the paths of the similar images and their similarity scores, with an option to visualize the query and result images

7.  Utility Functions (`src/utils.py`):
	* Contains helper functions, such as `load_config` for parsing the `config.yaml` file

## Setup Instructions
1.  Clone the Repository:
	```sh
	git clone https://github.com/shbzbarb/ImageSearch
	```

2.  Create and Activate Conda Environment:
	It's highly recommended to use Conda for environment management. You can create the environment using the provided `environment.yml` file:

	* Using `environment.yml` (Recommended):
    	This file contains the package versions used during development, including Conda and pip-installed packages.
    	```bash
    	conda env create -f environment.yml
    	conda activate image_search
    	```

## Execution and Usage
Make sure your Conda environment is activated before running any scripts. All scripts are located in the scripts/ directory

### Feature Extraction (`run_feature_extraction.py`)
Extracts feature vectors from all images in a specified directory and saves them as a .npy file, along with a .txt file containing the paths to the processed images
```sh
python scripts/run_feature_extraction.py -i path/to/your/image_dataset -o path/to/output/prefix_for_features
```

### Feature Extraction (`run_feature_extraction.py`)
Extracts feature vectors from all images in a specified directory and saves them as a .npy file, along with a .txt file containing the paths to the processed images
```sh
python scripts/run_feature_extraction.py -i path/to/your/image_dataset -o path/to/output/prefix_for_features
```
This will create `data/processed/my_images_vit_features.npy` and `data/processed/my_images_vit_paths.txt`

### Build Milvus Index (`build_milvus_index.py`)
Loads the extracted features and their corresponding image paths into a Milvus Lite database, creating a collection and a vector index for efficient searching
```sh
python scripts/build_milvus_index.py -f path/to/features.npy -p path/to/image_paths.txt
```

### Image Classification (`run_classification.py`)
Classifies a single input image and prints the predicted label and confidence score
```sh
python scripts/run_classification.py -i path/to/single/image.jpg
```

### Similarity Search (`run_similarity_search.py`)
Takes a query image, searches the Milvus database for similar images, and prints the results
```sh
python scripts/run_similarity_search.py -q path/to/query/image.jpg -k <number_of_results>
```
If -k is not specified, it defaults to the top_k value in `config.yaml` or 10


## Future Work & Known Issues
### Future Work:
- Implement batch processing for feature extraction for better efficiency on large datasets
- Support for more vector index types in Milvus and expose their parameters in config.yaml
- Add functionality for fine-tuning pre-trained models on custom datasets
- Develop a simple web interface (e.g., using Flask or Streamlit) for easier interaction
- Explore more advanced retrieval techniques (e.g., re-ranking, query expansion)

### Known Issues:
- Large image datasets may require significant time for initial feature extraction and indexing
- The current Milvus Lite setup is for local use; for production or larger scale, a full Milvus server deployment would be necessary


## References
* **Core Libraries & Frameworks:**
    * **PyTorch:** An open-source machine learning framework that accelerates the path from research prototyping to production deployment
        * [PyTorch Official Website](https://pytorch.org/)
    * **Hugging Face Transformers:** Provides thousands of pretrained models to perform tasks on different modalities such as text, vision, and audio
        * [Hugging Face Models Hub](https://huggingface.co/models)

* **Vector Database:**
    * **Milvus:** An open-source vector database built for AI applications, offering efficient similarity search and analytics for massive-scale vector data
        * [Milvus Official Website](https://milvus.io/)
    * **Pymilvus:** The Python SDK for Milvus
        * [Pymilvus on PyPI](https://pypi.org/project/pymilvus/)

* **Example Models:**
    * **Vision Transformer (ViT):** A model that applies the Transformer architecture, originally designed for text, directly to image classification.
        * Example: Google's ViT - [Paper "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"](https://arxiv.org/abs/2010.11929)

* **Example Dataset:**
    * **LIVE Image Quality Assessment Database:** A widely used dataset for image quality research
        * [LIVE Database Homepage](https://live.ece.utexas.edu/research/Quality/subjective.htm)