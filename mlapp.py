from flask import Flask, jsonify, request,render_template
#パッケージをインストールする場合は、パッケージ名.モジュール名にする
#ここではpredというモジュールの中からClassifierというクラスをインポートしている
from classification.pred import Classifier
from regression.pred import Regressor

#Flaskというクラスをインスタンス化
app = Flask(__name__)

#ClassifierとRegressorをインスタンス化。
#モデルによっては重みをロードするのに時間がかかるのではじめにロードしておく
CLASS_NUM = 5
pred_model_class = Classifier("weights/model_5class_dogs.pth",CLASS_NUM)
pred_model_reg = Regressor("weights/lgb_model.pkl")

@app.route("/")
def top():
    return render_template("top.html")

#画像分類
@app.route("/classification")
def class_input():
    return render_template("class_input.html")

#画像分類の予測の実行
@app.route("/classify", methods=['POST'])
def classify():
    uploaded_file = request.files['img_file']
    ans = pred_model_class.predict(uploaded_file)
    return jsonify({"result":ans})

#住宅価格予測
@app.route("/regression")
def reg_input():
    return render_template("reg_input.html")

#住宅価格の予測の実行
@app.route("/predict", methods=['POST'])
def predict():
    data = request.json
    result = pred_model_reg.predict(data)
    return jsonify({"result":result})

if __name__ == "__main__":
    app.run(host='0.0.0.0',debug=True)