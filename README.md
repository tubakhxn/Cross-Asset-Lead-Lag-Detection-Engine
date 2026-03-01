## dev/creator = tubakhxn

# Cross-Asset Lead-Lag Detection Engine
<img width="2385" height="1398" alt="image" src="https://github.com/user-attachments/assets/dc5a864c-a40b-4d59-a3e2-7639561e54bc" />

This project is a professional quant research tool for identifying dynamic lead-lag relationships and causality structures across multiple financial assets. It is designed for statistical arbitrage, quant research, and advanced financial analytics.

## What is this project?
- **Streamlit app** for interactive analysis and visualization
- Accepts CSV uploads or downloads price data from Yahoo Finance
- Computes log returns, rolling standardized returns, cross-correlation, Granger causality, DTW, and composite lead-lag strength
- Visualizes results with cinematic Plotly graphs: animated heatmap, 3D surface, network graph, lag distribution panel
- Full quant-themed UI with neon accents

## How to fork and run
1. **Fork this repository** on GitHub or copy the folder to your local machine.
2. **Install dependencies**:
   - Python 3.8+
   - `pip install streamlit pandas numpy plotly yfinance scipy statsmodels networkx`
3. **Run the app**:
   - `streamlit run app.py`
4. **Use the UI**:
   - Upload your CSV or download data from Yahoo Finance
   - Adjust analysis parameters in the sidebar
   - Explore interactive visualizations

## Main files
- `app.py` — The single-file Streamlit application containing all logic and UI
- `README.md` — This documentation file

---
For any questions or improvements, contact the creator: **tubakhxn**

