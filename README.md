# Stock Price Prediction Web Application

This project is a **Stock Price Prediction Web Application** built with **Flask**, **TensorFlow**, **Keras**, and **YFinance**. The application predicts the future stock prices based on historical data for **closing**, **opening**, and **high prices** of stocks. The model is pre-trained for **AAPL** (Apple) stock, but it can also be fine-tuned for any other stock ticker.

## Features

- **Stock Close Price Prediction**: Predict the future closing price of a stock.
- **Stock Opening Price Prediction**: Predict the future opening price of a stock.
- **Stock High Price Prediction**: Predict the future high price of a stock.
- **Stock Analysis**: Display stock data with various graphs for moving averages and stock prices.

### Graphs:
- **MA30 vs Close**
- **MA50 vs Close**
- **MA30 vs Open**
- **MA50 vs Open**
- **MA30 vs High**
- **MA50 vs High**

### Functional Flow:
1. **Prediction**:
   - Select the type of prediction (close, open, high).
   - Enter a stock ticker (default: `AAPL`).
   - Enter the number of days to predict (e.g., `30` days).
   - The application uses pre-trained models or fine-tunes models for non-`AAPL` tickers and predicts the stock prices.
   - The results are displayed in the form of tables and graphs.

2. **Stock Analysis**:
   - Fetches data for the selected ticker.
   - Displays the first 5 rows of data along with graphs for moving averages and stock prices.

---

## Technologies Used

- **Flask**: Web framework for backend API and rendering frontend.
- **TensorFlow / Keras**: For machine learning model development.
- **YFinance**: To fetch historical stock data.
- **HTML, CSS, JavaScript**: For frontend development.
- **Plotly**: For graph visualization.
- **Bootstrap**: For responsive UI.

---

