from torchvision import transforms
from PIL import Image   

#画像を変換する関数
def translation(img,device):
    #画像変換のルールを設定
    data_transforms= transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]) 
    #この時点でtorch.tensorというデータ形式になっている
    img_trans = data_transforms(img).unsqueeze(0)
    img_trans = img_trans.to(device)
    return img_trans

def url_to_image(img_object, device):
    #PillowでimageFileを開く
    img = Image.open(img_object)
    #画像の変換を行う
    img_trans = translation(img, device)
    return img_trans