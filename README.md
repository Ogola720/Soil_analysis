```markdown
# 🌱 Crop Recommendation System with SHAP Explanations

This project predicts the most suitable crop based on soil and environmental conditions 
using Machine Learning. It also explains **why** a certain crop was predicted via **SHAP (SHapley Additive Explanations).**

---

## 📂 Project Structure

```

crop-recommendation/
│── data/
│   └── Crop_recommendation.csv      # Original dataset
│   └── sample_input.csv             # Example soil/environmental input
│
│── models/
│   └── crop_model.pkl               # Trained ML model (RandomForest/LightGBM)
│   └── encoder.pkl                  # Label encoder for crop names
│
│── app.py                           # Streamlit web app
│── train_model.py                   # Script to train & save model
│── requirements.txt                 # Python dependencies
│── README.md                        # Project documentation

````

---

## ⚙️ Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone https://github.com/your-username/crop-recommendation.git
   cd crop-recommendation
````

2. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

   Example `requirements.txt`:

   ```
   streamlit
   pandas
   scikit-learn
   lightgbm
   shap
   matplotlib
   ```

3. **Train the Model (Optional)**

   ```bash
   python train_model.py
   ```

   This generates:

   * `models/crop_model.pkl`
   * `models/encoder.pkl`

4. **Run the Streamlit App**

   ```bash
   streamlit run app.py
   ```

5. **Upload Input Data**

   * Use the provided `sample_input.csv` or your own soil dataset.
   * The app outputs:

     * Recommended crop
     * SHAP-based explanation plot (feature contribution per sample)

---

## 📊 Example Prediction Flow

1. Upload `sample_input.csv` via the app.
2. Model predicts **Rice** for the first row.
3. SHAP plot shows:

   * `High rainfall` and `Moderate N` push the prediction towards Rice.
   * `Neutral pH` has minimal effect.

---

## 🧠 Tech Stack

* **Streamlit** → Simple, interactive UI.
* **LightGBM / RandomForest** → Crop prediction model.
* **SHAP** → Model interpretability.
* **Pandas + Matplotlib** → Data handling & visualization.

---

## 🔮 Future Work

* Integrate satellite imagery APIs (NASA, Sentinel Hub).
* Extend marketplace features for farmers.
* Deploy app to **Streamlit Cloud** or **Heroku**.

---

👨‍💻 **Author:** Peter
🎓 Dedan Kimathi University of Technology | Telecommunications & Information Engineering

```
