import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("solarpower_Test2.csv")

df = load_data()

# Sidebar for user input parameters
st.sidebar.header('User Input Parameters')

def user_input_features():
    distance_to_solar_noon = st.sidebar.slider('Distance to Solar Noon', min_value=0.0, max_value=3.14, value=1.57)
    temperature = st.sidebar.slider('Temperature (°C)', min_value=-30.0, max_value=80.0, value=20.0)
    wind_direction = st.sidebar.slider('Wind Direction (°)', min_value=0, max_value=360, value=180)
    wind_speed = st.sidebar.slider('Wind Speed (m/s)', min_value=0.0, max_value=20.0, value=5.0)
    sky_cover = st.sidebar.slider('Sky Cover (0-4)', min_value=0, max_value=4, value=2)
    visibility = st.sidebar.slider('Visibility (km)', min_value=0.0, max_value=50.0, value=10.0)
    humidity = st.sidebar.slider('Humidity (%)', min_value=0, max_value=100, value=50)
    average_wind_speed = st.sidebar.slider('Average Wind Speed (m/s)', min_value=0.0, max_value=20.0, value=5.0)
    average_pressure = st.sidebar.slider('Average Pressure (Hg in)', min_value=25.0, max_value=35.0, value=30.0)
    
    data = {
        'distance_to_solar_noon': distance_to_solar_noon,
        'temperature': temperature,
        'wind_direction': wind_direction,
        'wind_speed': wind_speed,
        'sky_cover': sky_cover,
        'visibility': visibility,
        'humidity': humidity,
        'average_wind_speed': average_wind_speed,
        'average_pressure': average_pressure
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Data Preprocessing
df.fillna(df.mean(), inplace=True)


# Split data
X = df.drop('power_generated', axis=1)
y = df['power_generated']


# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
scaled_input = scaler.transform(input_df)
prediction = model.predict(scaled_input)

# Display the prediction
st.write("### Predicted Power Generation (in Jules):")
st.write(f"{prediction[0]:.2f}")

# Model Performance
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

st.write("### Model Performance on Test Data:")
st.write(f"**Mean Squared Error (MSE):** {mse:.2f}")
st.write(f"**R-squared (R²):** {r2:.2f}")
st.write(f"**Mean Absolute Error (MAE):** {mae:.2f}")

# Visualization: Feature Importances
st.write("### Feature Importances:")
importances = model.feature_importances_
features = X.columns
importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

st.bar_chart(importance_df.set_index('Feature'))

# Visualization: Correlation Matrix
st.write("### Best Model is Ramdom Forest Regression:")


plt.figure(figsize=(10, 6))
sns.barplot(x=model.feature_importances_, y=X.columns, palette='viridis')
plt.title('Random Forest Feature Importance')
st.pyplot(plt)


