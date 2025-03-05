## AUTHOR : NATHISHWAR

# **Skillcraft_intern_projects**

#SCT_ML_1 TASK
# Linear Regression: House Price Prediction

## Overview
This project implements a **Linear Regression model** to predict house prices based on features such as **square footage, number of bedrooms, and number of bathrooms** using the Kaggle dataset.

## Dataset
The dataset contains real estate data, including:
- **Square Footage** (`sqft`)
- **Number of Bedrooms** (`bedrooms`)
- **Number of Bathrooms** (`bathrooms`)
- **House Price** (`price` - target variable)

Ensure that you download and extract the dataset before running the script.

## Installation
To run this project, install the required dependencies:

```bash
pip install numpy pandas scikit-learn matplotlib
```

## How It Works
1. **Load Data**: Reads the dataset into a pandas DataFrame.
2. **Preprocessing**:
   - Handles missing values (if any).
   - Splits the dataset into **features (X)** and **target variable (y)**.
   - Splits data into **training (80%)** and **testing (20%)** sets.
3. **Training the Model**:
   - Fits a **Linear Regression** model using `scikit-learn`.
4. **Evaluation**:
   - Predicts house prices for the test set.
   - Computes **Mean Absolute Error (MAE)** and **R² Score**.
5. **Visualization**:
   - Plots predicted vs. actual prices.

## Running the Script
Modify the `dataset_path` variable in `house_price_prediction.py` and run:

```bash
python house_price_prediction.py
```

## Expected Output
The script prints the model's performance metrics:

```
Mean Absolute Error: 24500.23  # Example output (varies)
R² Score: 0.82
```

## Future Enhancements
- Add more features like **location** and **year built**.
- Use **Polynomial Regression** for better accuracy.
- Implement **Deep Learning** models (e.g., Neural Networks) for improved predictions.
------------------------------------------------------------------------------------------

#SCT_ML_2 TASK 
# **Customer Segmentation using K-Means Clustering**  

## **Overview**  
This project applies **K-Means clustering** to segment customers of a retail store based on their purchase behavior. The dataset includes customer details such as **age, gender, annual income, and spending score**.  

## **Dataset**  
- **CustomerID**: Unique ID for each customer  
- **Gender**: Male or Female  
- **Age**: Customer's age  
- **Annual Income (k$)**: Annual income in thousands  
- **Spending Score (1-100)**: Customer spending behavior score  

## **Requirements**  
- Python 3.x  
- Libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `sklearn`  

## **Steps to Run**  
1. Install dependencies:  
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```  
2. Run the Python script to load the dataset, preprocess it, and apply K-Means clustering.  

## **Results**  
- The model groups customers into clusters based on their purchasing patterns.  
- Visualizations help understand different customer segments.  

## **Use Cases**  
- Targeted marketing  
- Personalized promotions  
- Customer behavior analysis  
-------------------------------------------------------------------------------------------------------------------------
#SCT_ML_3 TASK
# SVM Image Classifier: Cats vs. Dogs

## Overview
This project implements a **Support Vector Machine (SVM)** classifier to distinguish between images of **cats** and **dogs** using the Kaggle dataset.

## Dataset
The dataset consists of images labeled as either **cats** or **dogs**. Ensure that you download and extract the dataset before running the script.

- **Dataset Path:** `path_to_kaggle_dataset/train/`
- **Categories:** `cat/` and `dog/`

## Installation
To run this project, install the required dependencies:

```bash
pip install numpy opencv-python scikit-learn
```

## How It Works
1. **Load Images**: The script reads images from the dataset folder.
2. **Preprocessing**:
   - Converts images to grayscale.
   - Resizes them to `64x64` pixels.
   - Flattens images into feature vectors.
   - Normalizes pixel values.
3. **Training the SVM**:
   - Uses an **RBF kernel** for classification.
   - Trains on an 80-20 train-test split.
4. **Evaluation**:
   - Predicts labels for test images.
   - Computes accuracy using `accuracy_score`.

## Running the Script
Modify the `dataset_path` variable in `svm_classifier.py` and run:

```bash
python svm_classifier.py
```

## Expected Output
The script prints the model's accuracy on the test set:

```
Test Accuracy: 0.85  # Example output (varies)
```

## Future Enhancements
- Use **HOG (Histogram of Oriented Gradients)** for better feature extraction.
- Optimize SVM parameters using **GridSearchCV**.
- Implement a **CNN model** for improved accuracy.

---------------------------------------------------------------------------------

#SCT_ML_4 TASK 

# **Hand Gesture Dataset Collection**  

This script captures hand gesture images using OpenCV and saves them into labeled folders for training a gesture recognition model.  

## **How to Use**  
1. **Install dependencies** (if not installed):  
   ```bash
   pip install opencv-python
   ```  
2. **Run the script**:  
   ```bash
   python collect_gesture.py
   ```  
3. **Enter the gesture name** when prompted (e.g., "thumbs_up").  
4. The script will **capture 1,000 images** and save them in `dataset/{gesture_name}/`.  
5. **Press 'q' to quit** anytime.  

## **Dataset Structure**  
```
dataset/
│── thumbs_up/
│   ├── 0.jpg
│   ├── 1.jpg
│   ├── ...
│── fist/
│   ├── 0.jpg
│   ├── 1.jpg
│   ├── ...
```

## **Next Steps**  
- Collect images for multiple gestures.  
- Perform data augmentation to improve model accuracy.  
- Train a CNN-based model for gesture recognition.  

