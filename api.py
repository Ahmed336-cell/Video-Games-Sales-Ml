from flask import Flask, request, render_template, redirect
from app import Predict

cp = Predict()
app = Flask(__name__)  ## __name__= current file name  (main)


@app.route("/")  ## page name
def predict_html():
    return render_template("predict.html")


@app.route("/predict")  ## page name
def predict():
    model = request.args["model"]
    Platform = request.args["Platform"]
    Genre = request.args["Genre"]
    NA_Sales = float(request.args["NA_Sales"])
    EU_Sales = float(request.args["EU_Sales"])
    JP_Sales = float(request.args["JP_Sales"])
    Other_Sales = float(request.args["Other_Sales"])

    test = [model, Platform, Genre, NA_Sales, EU_Sales, JP_Sales, Other_Sales]
    label = cp.pred(test)
    return render_template("result.html", message=label)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
