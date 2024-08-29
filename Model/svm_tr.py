from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

years = list(range(1950, 2021))
renewable_energy = [4250, 4492, 4733, 4975, 5217, 5458, 5700, 5942, 6183, 6425, 6667, 6908, 7150, 7392, 7633, 7875, 8117, 8358, 8600, 8842, 9083, 9425, 9767, 10108, 10450, 10792, 11133, 11475, 11817, 12158, 12500, 12942, 13383, 13825, 14267, 14708, 15150, 15592, 16033, 16475, 16917, 17917, 18917, 19917, 20917, 21917, 22917, 23917, 24917, 25917, 26917, 28917, 30917, 32917, 34917, 36917, 38917, 40917, 42917, 44917, 46917, 51917, 56917, 61917, 66917, 71917, 76917, 81917, 86917, 91917, 96917]

data = {'Year': years, 'Renewable Energy (MW)': renewable_energy}
data = pd.DataFrame(data)

train_data = data[data['Year'] <= 2015]
test_data = data[data['Year'] > 2015]

# Prepare the data for SVM
X_train = train_data[['Year']]
y_train = train_data['Renewable Energy (MW)']]

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Train the SVM model
svm_model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
svm_model.fit(X_train_scaled, y_train)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    year = None
    if request.method == 'POST':
        year = int(request.form['year'])
        if year < 1950 or year > 2050:
            prediction = "Year out of range. Please select a year between 1950 and 2050."
        else:
            if year <= 2020:
                # Use the dataset to get the actual value
                prediction = data[data['Year'] == year]['Renewable Energy (MW)'].values[0]
            else:
                # Use the SVM model to predict the value
                future_years = np.array([[year]])
                future_years_scaled = scaler.transform(future_years)
                prediction = svm_model.predict(future_years_scaled)[0]

    return render_template('index.html', year=year, prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)