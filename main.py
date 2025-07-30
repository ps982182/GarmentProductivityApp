from flask import Flask, render_template, request
import numpy as np
import pickle
import io
import base64
import matplotlib.pyplot as plt

app = Flask(__name__)
model = pickle.load(open('model_xgb.pkl','rb'))

@app.route("/")
@app.route("/predict")
def form():
    return render_template("predict.html")

@app.route("/pred", methods=["POST"])
def predict():
    try:
        # Collect input data
        quarter = request.form['quarter']
        department = request.form['department']
        day = request.form['day']
        team = request.form['team']
        targeted_productivity = request.form['targeted_productivity']
        smv = request.form['smv']
        over_time = request.form['over_time']
        incentive = request.form['incentive']
        idle_time = request.form['idle_time']
        idle_men = request.form['idle_men']
        no_of_style_change = request.form['no_of_style_change']
        no_of_workers = request.form['no_of_workers']
        month = request.form['month']

        total = [[int(quarter), int(department), int(day), int(team),
                  float(targeted_productivity), float(smv), int(over_time), int(incentive),
                  float(idle_time), int(idle_men), int(no_of_style_change), float(no_of_workers), int(month)]]

        prediction = model.predict(total)[0]
        if prediction <= 0.3:
            text = '🟥 The employee is Averagely Productive.'
        elif 0.3 < prediction <= 0.8:
            text = '🟨 The employee is Medium Productive.'
        else:
            text = '🟩 The employee is Highly Productive.'

        graphs = []
        labels = ['Targeted Productivity', 'SMV', 'Over Time', 'Idle Time']
        values = [float(targeted_productivity), float(smv), float(over_time), float(idle_time)]

        for kind in ['bar', 'scatter', 'line', 'pie']:
            plt.figure()
            if kind == 'bar':
                plt.bar(labels, values)
                plt.title('Bar Chart')
            elif kind == 'scatter':
                plt.scatter(labels, values)
                plt.title('Scatter Plot')
            elif kind == 'line':
                plt.plot(labels, values, marker='o')
                plt.title('Line Plot')
            elif kind == 'pie':
                plt.pie(values, labels=labels, autopct='%1.1f%%')
                plt.title('Pie Chart')
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            graphs.append(base64.b64encode(buf.read()).decode('utf-8'))
            plt.close()

        return render_template("submit.html", prediction_text=text, graphs=graphs)
    except Exception as e:
        return f"❌ Error occurred: {e}"

if __name__ == "__main__":
    app.run(debug=True)
