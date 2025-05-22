# PRODIGY_ML_TASK2
Customer Segmentation using K-Means Clustering on mall data. Started with basic features and advanced to include age, feature scaling, silhouette analysis, PCA, and visual insights. Helps identify customer groups for targeted marketing strategies.

# Mall Customer Segmentation – Advanced K-Means Clustering

## Overview

This project is part of my internship at **Prodigy InfoTech** and focuses on performing **customer segmentation** for a mall using the **K-Means Clustering** algorithm.

I started with a basic model and gradually upgraded it to an advanced version by adding meaningful features, better evaluation techniques, and insightful visualizations.

---

## Basic Features Implemented

- Imported and explored the `Mall_Customers.csv` dataset
- Selected **Annual Income** and **Spending Score** for clustering
- Applied **K-Means** with a fixed number of clusters
- Plotted simple scatter visualizations

---

## Advanced Features Added

- Included **Age** to enhance clustering relevance  
- Applied **StandardScaler** to scale features  
- Used **Silhouette Score Analysis** to determine optimal cluster count  
- Implemented **PCA** for 2D visualization of high-dimensional data  
- Created **pairwise scatterplots** for deeper pattern discovery  
- Replaced seaborn with **pure matplotlib** for broader compatibility  
- Generated **automated business insights** for each customer group

---

## File Structure

- `mall_customer_segmentation.py` – Basic clustering version  
- `mall_customer_segmentation_advanced.py` – Final advanced model with added features  
- `README.md` – Project documentation

---

## How to Run

1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)  
2. Place `Mall_Customers.csv` in the same directory as your `.py` file  
3. Install required libraries:
   ```bash
   pip install pandas numpy matplotlib scikit-learn
````

4. Run the script:

   ```bash
   python mall_customer_segmentation_advanced.py
   ```

## Tools & Technologies

* Python
* Pandas, NumPy
* Scikit-learn
* Matplotlib
* K-Means Clustering
* PCA, Silhouette Score

---

## Outcome

This project helped me:

* Understand how unsupervised learning works in the real world
* Learn model evaluation and data scaling techniques
* Convert basic clustering into a robust, business-ready analysis

---

## 📽️ Demo Video
🎥 [Watch the project walkthrough video on LinkedIn]- www.linkedin.com/in/yada-rajeshwari-022b8530b

## Credits
Internship: Prodigy InfoTech
Dataset: [Mall Customers Dataset - Kaggle](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)



