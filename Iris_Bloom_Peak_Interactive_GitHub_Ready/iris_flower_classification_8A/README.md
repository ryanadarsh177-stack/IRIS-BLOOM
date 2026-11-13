# Iris Bloom | Interactive Flower Lab

A floral, animated Streamlit experience for exploring the classic Iris classification dataset.

## Features
- Animated botanical hero, floating petals, soft pastel glass cards and responsive layout.
- Live classification with K-Nearest Neighbors and Logistic Regression.
- Measurement sliders and one-click flower presets.
- Interactive Plotly scatter plot with species filters and selectable dimensions.
- Model greenhouse with accuracy, confusion matrix and classification report.
- Filterable dataset explorer and CSV download.
- In-app star rating and feedback form.

## Deploy on Streamlit Community Cloud
1. Create a GitHub repository.
2. Upload the **contents** of this folder to the repository root.
3. Open Streamlit Community Cloud and choose **Create app**.
4. Select your repository, branch (`main`), and main file path (`app.py`).
5. Deploy.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

The app reads `data/iris.csv` and `assets/iris_luminous.svg` using paths relative to `app.py`.
