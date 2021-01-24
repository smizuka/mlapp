import numpy as np
import pickle

class Regressor:

    def __init__(self, model_name):
        self.model = pickle.load(open(model_name, 'rb'))

    #辞書型で入力される
    def predict(self, dic):
        #dic.values()でvalueだけ取得。valueはすべて文字列なので整数に変換
        map_obj = map(lambda x : int(x), dic.values())
        #mapobjectのままだと使えないのでリスト型にする
        data = list(map_obj)
        #リストをnumpyarrayにしたあと１次元増やす。(レコード数、カラム数)の行列で入力されるため。
        x = np.array(data)[np.newaxis,:]
        pred = self.model.predict(x)
        #出力はnumpyarrayなのでリスト型にしてから小数点以下切り捨て
        pred = round(pred.tolist()[0])
        #formatで3コンマ区切りにする
        return "{:,}".format(pred)+"ドル"