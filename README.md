
# 🌱 Crop Recommendation System with SHAP

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python" alt="Python Version">
  <img src="https://img.shields.io/badge/Framework-Streamlit-red?style=for-the-badge&logo=streamlit" alt="Streamlit Version">
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License">
</p>

<p align="center">
  A smart farming tool that recommends the best crop for your land and tells you <em>why</em>.
  <br />
  <a href="#-about-the-project"><strong>Explore the docs »</strong></a>
  <br />
  <br />
  <a href="https://ogola720-soil-analysis-app-sbsc5p.streamlit.app/">View Demo</a>
</p>

<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#-about-the-project">About The Project</a></li>
    <li><a href="#-features">Features</a></li>
    <li><a href="#-built-with">Built With</a></li>
    <li><a href="#-getting-started">Getting Started</a></li>
    <li><a href="#-usage">Usage</a></li>
    <li><a href="#-project-structure">Project Structure</a></li>
    <li><a href="#-roadmap">Roadmap</a></li>
    <li><a href="#-contact">Contact</a></li>
  </ol>
</details>

---

## 📖 About The Project



This project provides an intelligent crop recommendation system powered by Machine Learning. Based on key soil and environmental factors—such as Nitrogen, Phosphorus, Potassium levels, temperature, humidity, pH, and rainfall—it predicts the most suitable crop to cultivate.

What sets this project apart is its **explainability**. Using **SHAP (SHapley Additive Explanations)**, it doesn't just give you a recommendation; it shows you exactly which factors were most influential in the decision, making the AI's reasoning transparent and trustworthy.

---

## ✨ Features

* **📈 Intelligent Crop Prediction:** Recommends the ideal crop using a trained LightGBM/Random Forest model.
* **🔍 Explainable AI (XAI):** Generates SHAP plots to explain the contribution of each feature to the prediction.
* **📁 CSV Upload:** Upload a `.csv` file with your soil and weather data for instant predictions.
* **🌐 Interactive Web App:** A simple and user-friendly interface built with Streamlit.

---

## 🛠️ Built With

This project leverages a powerful stack of open-source libraries:

* **Backend & ML:**
    * [Scikit-learn](https://scikit-learn.org/)
    * [LightGBM](https://lightgbm.readthedocs.io/en/latest/)
    * [SHAP](https://shap.readthedocs.io/en/latest/)
    * [Pandas](https://pandas.pydata.org/)
* **Frontend & Visualization:**
    * [Streamlit](https://streamlit.io/)
    * [Matplotlib](https://matplotlib.org/)

---

## 🚀 Getting Started

Follow these steps to get a local copy up and running.

### Prerequisites

Make sure you have **Python 3.9 or higher** installed on your system.

### Installation

1.  **Clone the Repository**
    ```sh
    git clone [https://github.com/](https://github.com/)[YOUR_USERNAME]/crop-recommendation.git
    cd crop-recommendation
    ```

2.  **Create a Virtual Environment (Recommended)**
    ```sh
    # For Unix/macOS
    python3 -m venv .venv
    source .venv/bin/activate

    # For Windows
    python -m venv .venv
    .venv\Scripts\activate
    ```

3.  **Install Dependencies**
    ```sh
    pip install -r requirements.txt
    ```

4.  **Train the Model (Optional)**
    The repository includes pre-trained models. However, if you want to retrain the model with new data, simply run:
    ```sh
    python train_model.py
    ```
    This will regenerate `models/crop_model.pkl` and `models/encoder.pkl`.

---

## 🖥️ Usage

Once the setup is complete, you can launch the interactive web application.

1.  **Run the Streamlit App**
    ```sh
    streamlit run app.py
    ```
    Your web browser will open a new tab with the application running locally.

2.  **Upload Your Data**
    * Use the sidebar to upload a `.csv` file containing your soil and environmental data.
    * You can use the `data/sample_input.csv` file for a quick demonstration.

3.  **Get Recommendations & Explanations**
    * The app will display the recommended crop for each row of data in your input file.
    * For each prediction, a **SHAP plot** is generated, visualizing how each factor (e.g., high rainfall, low pH) contributed to the final decision.



**Example Flow:**
1.  Upload `sample_input.csv` via the app.
2.  The model predicts **"Rice"** for the first row.
3.  The SHAP plot shows that `High rainfall` and `Moderate N` values strongly pushed the prediction towards Rice, while a `Neutral pH` had a minimal effect.

---

## 📂 Project Structure

````

crop-recommendation/
├── data/
│   ├── Crop\_recommendation.csv  \# The full dataset used for training
│   └── sample\_input.csv         \# An example file for predictions
│
├── models/
│   ├── crop\_model.pkl           \# Pre-trained ML model (RandomForest/LightGBM)
│   └── encoder.pkl              \# Label encoder for crop names
│
├── app.py                       \# The main Streamlit application file
├── train\_model.py               \# Script to train and save the model
├── requirements.txt             \# Required Python packages for pip
└── README.md                    \# You are here\!


````
## 🔮 Roadmap

-   [ ] Integrate with weather and soil data APIs to fetch real-time data.
-   [ ] Extend with marketplace features for farmers.
-   [ ] Deploy the application to Streamlit Cloud or Heroku.
-   [ ] Add summary SHAP plots for batch predictions.


---

## 👨‍💻 Contact

**Peter Ogola** - [ayienga.peter@gmail.com]
** Brian Muthengi** -[brianmuthengi91@gmail.com]
