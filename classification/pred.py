import torch
from torchvision import models
#同じパッケージ内のモジュールを使う際は.utilsのように先頭に.をつける。
#utilsというモジュールからtranslationという関数とurl_to_imageという関数をインポートしている
from .utils import translation, url_to_image

def create_model(device, class_num):
    #ベースとなるモデルの構造をロード
    model = models.resnet18(pretrained=False)
    #出力層の差し替え
    num_feature = model.fc.in_features
    model.fc = torch.nn.Linear(num_feature, class_num)
    return model.to(device)

class Classifier:

    def __init__(self, model_name, class_num):
        #デバイスの自動判別（CPUとGPUではデータの形式が異なるため）
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #モデルの重みをロードする
        load_weights = torch.load(model_name, map_location = torch.device('cpu'))
        #モデルをデバイスに応じて変換
        model = create_model(self.device, class_num)
        #モデルに重みをセットする
        model.load_state_dict(load_weights)
        self.model = model
    
    #モデルの出力する数値をラベルに変換する
    def convert(self, number):
        labels = labels = {
            0:'american_bulldog',
            1:'american_pit_bull_terrier',
            2:'basset_hound',
            3:'beagle',
            4:'chihuahua'
        }
        return labels[number]

    #クラス内の関数には（）内にselfをつける
    def predict(self, img_object):
        #画像の変換処理
        img = url_to_image(img_object, self.device)
        #推論モードにする
        self.model.eval()
        #推論なのでパラメータの保存を止める
        with torch.no_grad():
            outputs = self.model(img)
            _, preds = torch.max(outputs, 1)

        #予測された数値をラベルに変換
        ans = self.convert( preds.item())

        return ans