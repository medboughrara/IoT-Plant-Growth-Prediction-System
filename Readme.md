# 🌱 IoT Plant Growth Prediction System

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://python.org)
[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.24.0-FF4B4B.svg)](https://streamlit.io)

An advanced machine learning system for predicting plant growth patterns using IoT greenhouse sensor data. This project enables real-time monitoring and classification of plant health conditions based on comprehensive growth parameters.

## 📋 Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Dashboard](#dashboard)
- [Contributing](#contributing)
- [License](#license)

## 🔍 Overview

This project implements a machine learning system for predicting plant growth patterns using IoT sensor data in greenhouses. The system uses various environmental and plant metrics to classify and predict plant growth patterns, providing real-time monitoring and insights through an interactive dashboard.

### Key Benefits
- **Real-time monitoring** of plant growth metrics
- **Predictive analytics** for plant health assessment
- **Interactive dashboard** for data visualization
- **Automated IoT data processing** for continuous monitoring

## 📁 Project Structure

```
IoT-Plant-Growth-Prediction-System/
├── data/
│   └── Advanced_IoT_Dataset.csv
├── src/
│   ├── models/
│   │   └── plant_growth_predictor.py
│   └── utils/
│       ├── iot_monitor.py
│       └── visualization.py
├── dashboard.py
├── train.py
├── requirements.txt
└── README.md
```

## ✨ Features

### 🤖 Machine Learning Components
- Advanced plant growth prediction model
- Multiple algorithm comparison (Random Forest, Gradient Boosting, SVM)
- Automated model selection and hyperparameter tuning
- Feature importance analysis and visualization

### 📊 Interactive Dashboard
- Real-time monitoring of plant metrics
- Environmental condition visualization
- Growth rate predictions
- Historical data analysis
- Statistical insights

### � Utility Functions
- IoT sensor data processing
- Data visualization and plotting
- Real-time monitoring
- Batch prediction capabilities

## 📊 Dataset

The **Advanced_IoT_Dataset.csv** contains 30,000 records with 14 columns:

### Plant Growth Parameters
| Feature | Description | Type |
|---------|-------------|------|
| ACHP | Average chlorophyll content | Float |
| PHR | Plant height growth rate | Float |
| AWWGV | Average wet weight of vegetative growth | Float |
| ALAP | Average leaf area | Float |
| ANPL | Average number of leaves | Float |
| ARD | Average root diameter | Float |
| ADWR | Average dry weight of roots | Float |
| PDMVG | Percentage of dry matter (vegetative) | Float |
| ARL | Average root length | Float |
| AWWR | Average wet weight of roots | Float |
| ADWV | Average dry weight of vegetative parts | Float |
| PDMRG | Percentage of dry matter (roots) | Float |
| Class | Plant classification category | Object |

### Data Source
- **Institution**: Department of Computer Science, Tikrit University, Iraq
- **Research Period**: 2023-2024
- **Principal Investigator**: Mohammed Ismail Lifta
- **Supervisor**: Prof. Wisam Dawood Abdullah
- **Collection Method**: IoT sensors in controlled greenhouse environment

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/medboughrara/IoT-Plant-Growth-Prediction-System.git
cd IoT-Plant-Growth-Prediction-System
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
.\venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 🚀 Usage

1. Train the model:
```bash
python train.py
```
This will:
- Load and preprocess the dataset
- Train the machine learning model
- Perform feature analysis
- Save the trained model

2. Run the dashboard:
```bash
streamlit run dashboard.py
```
This will start the interactive dashboard where you can:
- Monitor real-time plant metrics
- View environmental conditions
- Make growth predictions
- Analyze historical data

## � Dataset

The dataset (`Advanced_IoT_Dataset.csv`) contains plant growth measurements including:

- Average chlorophyll content
- Plant height rate
- Growth vegetative wet weight
- Leaf area
- Number of leaves
- Root measurements
- Growth rates
- Environmental conditions

## �️ Dashboard

The Streamlit dashboard provides:

### Monitoring Features
- Environmental conditions tracking
- Growth rate visualization
- Real-time predictions
- Historical data analysis

### Interactive Components
- Date range selection
- Growth prediction form
- Statistical summaries
- Interactive plots

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## � Authors

- **Mohamed Boughrara** - *Initial work* - [medboughrara](https://github.com/medboughrara)

### Research Collaboration
For academic collaborations or research partnerships, please contact:
- **Primary Researcher**: Mohammed Ismail Lifta
- **Academic Supervisor**: Prof. Wisam Dawood Abdullah
- **Institution**: Tikrit University, Iraq

---

**Made with ❤️ for sustainable agriculture and IoT innovation**

*Star ⭐ this repository if you find it useful!*