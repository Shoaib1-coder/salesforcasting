# 📈 Sales Forecasting App

An interactive web application built using **Streamlit** and **Facebook Prophet** that allows users to upload their sales data and generate time-series forecasts with interactive visualizations.

---

## 🚀 Features

- Upload CSV or Excel sales data.
- Automatically detects date and sales columns.
- Smooths data and applies optional log transform.
- Trains a Prophet model or uses a pre-trained one.
- Plots actual vs predicted sales using Plotly.
- Displays forecast table and original data.
- Supports monthly forecasts up to 60 months.

---

## 🔧 Setup Instructions

### 📥 1. Clone the Repository

```bash
git clone https://github.com/shoaib1-coder/salesforecasting.git
cd sales-forecasting-app
```

### 🐍 2. Download & Install Anaconda

If you don’t have Anaconda installed, [download it here](https://www.anaconda.com/products/distribution) and install for your OS.

### 🛠️ 3. Create Environment Using Anaconda Prompt

Open **Anaconda Prompt**, then:

```bash
conda create --name sales-forecast python=3.11
conda activate sales-forecast
pip install -r requirements.txt
```

---

## ▶️ Run the App

```bash
streamlit run app.py
```

Open your browser at: [http://localhost:8501](http://localhost:8501)

---

## 📂 File Structure

```
├── app.py                 # Main Streamlit application
├── preprocessdata.csv     # Default dataset used when no file is uploaded
├── salesforecast.pkl      # Optional: pretrained Prophet model
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

---

## 📊 Sample Screenshot

![Sales Forecasting UI](https://github.com/shoaib1-coder/sales-forecasting-app/raw/main/screenshot.png)

---

## 👨‍💻 Author

**Muhammad Shoaib Sattar**  
📧 Email: [mshoaib3393@gmail.com](mailto:mshoaib3393@gmail.com)  
🔗 LinkedIn: [linkedin.com/in/shoaib-0b64a2204](https://www.linkedin.com/in/shoaib-0b64a2204)  
💻 GitHub: [github.com/shoaib1-coder](https://github.com/shoaib1-coder)

---

## 📝 License

This project is licensed under the MIT License. You can freely use, modify, and distribute it with attribution.

---

## 🙏 Acknowledgements

- [Streamlit](https://streamlit.io/)
- [Facebook Prophet](https://facebook.github.io/prophet/)
- [Plotly](https://plotly.com/)
