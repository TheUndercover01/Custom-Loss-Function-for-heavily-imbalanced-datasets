# Medical Image Segmentation for Tumor Detection Using a Custom Loss Function to Address Severe Class Imbalance

This repository implements a deep learning solution to address the challenges of extreme class imbalance in medical image segmentation, particularly for tumor detection in PET and CT scans. The solution incorporates advanced segmentation techniques, including custom loss functions, and demonstrates a significant improvement in model performance.

## **Project Overview**
Medical image segmentation plays a crucial role in diagnosing and monitoring diseases like cancer. However, it often encounters challenges like extreme class imbalance, where tumor pixels make up less than 0.01% of the total dataset. This project introduces a novel approach to tumor detection by overcoming class imbalance through custom loss functions and leveraging a deep learning-based model architecture.

### **Key Achievements**
- Developed a novel and efficient segmentation solution for tumor detection in PET and CT scans.
- Processed a large medical image dataset containing over **30,000** images.
- Successfully addressed the extreme class imbalance problem by implementing a **Binary Weighted Focal Tversky Loss** function.
- Improved the model's performance, resulting in an enhanced **DICE Score** from **0.41 to 0.656**.
- Achieved the highest performance among **23** different model configurations tested.

## **Problem Statement**
Medical image segmentation for tumor detection presents the following challenges:
- **Extreme Class Imbalance**: Tumor regions represent less than 0.01% of the total dataset, making it difficult for traditional models to detect and learn tumor features effectively.
- **Low Detection Accuracy**: Rare and critical medical conditions often lead to poor detection accuracy, which is detrimental to patient outcomes.

## **Technical Approach**

### Binary Weighted Focal Tversky Loss

Address extreme class imbalance in medical image segmentation, particularly in scenarios with rare positive classes (e.g., tumor detection).

#### Loss Calculation
- **Sigmoid Transformation**: Converts raw model outputs to probabilities
- **Class Imbalance Weighting**: Dynamically adjusts weights based on positive/negative sample ratio
- **Tversky Index**: Measures overlap between predicted and ground truth segmentations
- **Focal Modification**: Enhances focus on hard-to-classify samples

#### Parameters
- `alpha`: False Positive weight (default: 0.5)
- `beta`: False Negative weight (default: 0.5)
- `gamma`: Focal loss intensity (default: 1)
- `epsilon`: Prevents division by zero (default: 1e-7)

#### Weighted Loss Combination
Final loss combines:
- Binary Cross-Entropy (60%)
- Binary Weighted Focal Tversky Loss (40%)
- Dice Loss (60%)

#### Advantages
- Handles severe class imbalance
- Reduces false positives/negatives
- Improves segmentation accuracy in medical imaging

#### Implementation
```python
total_vloss = (
    0.6 * BCELoss + 
    0.4 * bin_weighted_focal_tversky + 
    0.6 * dice_loss
)
```
### **Model Architecture**
The deep learning model used was **TransUNet**.
### **Model Performance**
- **Initial DICE Score**: 0.41
- **Achieved DICE Score**: 0.656
- **Best Performing Model**: Among 23 models tested, this solution consistently ranked highest in tumor detection accuracy.

## **Technologies Used**
- **PyTorch**: Deep learning framework for model implementation and training.
- **Segmentation Models PyTorch (SMP)**: Pre-trained segmentation models library to speed up the development of the solution.
- **NumPy**: Used for handling large numerical datasets and array manipulations.
- **MedPy**: Medical image processing library, specifically for evaluating segmentation models with medical metrics.
  
## **Research Collaboration**
This project was conducted in collaboration with the **Image and Speech Processing Lab** at **NIT Karnataka**, India. The research period spans from **October 2023 to February 2024** under the supervision of **Assoc. Prof. Jeny Rajan**.

## **Repository Structure**

The repository is organized into the following files:

- **`config.py`**: Contains configuration settings such as model hyperparameters, dataset paths, and training parameters.
- **`loss.py`**: Implementation of the custom **Binary Weighted Focal Tversky Loss** function.
- **`training.py`**: Contains the code for model training, including data loading, optimization, and training loop.
- **`test.py`**: Used for evaluating the model performance on test data, generating metrics like DICE Score and IoU.
- **`main.py`**: The main entry point of the repository, which integrates the entire pipeline for training and evaluation.

## **Getting Started**

Follow the steps below to set up and run the project:

### **1. Clone the Repository**
```bash
git clone https://github.com/yourusername/medical-image-segmentation.git
cd medical-image-segmentation
```
### **2. Install Required Dependencies**
Ensure you have Python 3.8+ and pip installed. Then install the necessary packages:
```bash
pip install -r requirements.txt
```

### **3. Prepare Your Dataset**
Prepare a medical imaging dataset (e.g., CT or PET scan images). Ensure that the dataset is in the required format (typically images in `.nii`, `.dcm`, or `.png` formats). Update the dataset paths in the `config.py` file.
### **4. Train the Model**
Run the `main.py` script to start training:
```bash
python main.py
```
This will load the dataset, configure the model, and start the training process.
### **5. Evaluate the Model**
After training, you can evaluate the model's performance using the `test.py` script:
```bash
python test.py
```
This will generate performance metrics, including the DICE Score, IoU, and others, to evaluate the model's effectiveness.

## Future Work

Future developments for this project include:

- Multi-Class Segmentation: Extend the model to handle multi-class segmentation problems, such as detecting multiple types of tumors or abnormalities.
- Exploring Advanced Encoder Architectures: Experiment with other encoder architectures like EfficientNet or Vision Transformers (ViT) to further improve performance.
- Generalizing Loss Function: Modify the custom loss function to be more applicable to a variety of medical imaging scenarios, such as organ segmentation or detection of other abnormalities.

## License

This repository is licensed under the MIT License.

## Contact

For any questions or inquiries, feel free to reach out at:

Email: [Ayush Deshmukh NITK email](mailto:aad.211it014@nitk.edu.in)\
LinkedIn: [Ayush Deshmukh](https://www.linkedin.com/in/ayush-deshmukh-64b2b5226/)