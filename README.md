# Image Analysis and Similarity Search Project

## Codes Coming Soon...

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
The idea behind this project is to use existing open-source resources for image analysis tasks specifically image feature extraction, image classification and image similarity search. The project leverages deep learning models to understand image content and a vector database for efficient retrieval of similar images.

## Core Functionalities

1.  **Image Classification:** Predicts and assigns a descriptive label to an input image using pre-trained models
2.  **Image Feature Extraction:** Converts images into dense numerical vectors. These vectors represent the semantic features of the images, allowing them to be compared
3.  **Image Similarity Search:** Finds images within a collection that are visually similar to a query image. This is achieved by:
    * Extracting the feature vector of the query image
    * Comparing this vector against a database of pre-computed feature vectors (using Milvus Lite)
    * Retrieving the images corresponding to the closest vectors in the feature space

## Technology Stack
The project utilizes the following core libraries and technologies:

* **Deep Learning Framework:**
    * **PyTorch:** Used for building and running the deep learning models
    * **CUDA Support:** Enabled for GPU acceleration
* **Model Hub & Utilities:**
    * **Transformers:** Provides access to state-of-the-art pre-trained models for image classification and feature extraction tasks
* **Vector Database:**
    * **Milvus Lite (`pymilvus`):** An embedded vector database used for efficient storage and fast similarity searching of the extracted image feature vectors
