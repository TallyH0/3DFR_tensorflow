from model import Net_3DFR
from data_loader import Data_loader_sequence

img_h, img_w, img_c = 240, 360, 1

loaders = []

dir_datas = 'F:/Data/CDNet2014/dataset/baseline/pedestrians'

dir_model = 'model_CDNet2014_baseline'
dir_output = 'outputs'


loaders = Data_loader_sequence(dir_datas, img_h, img_w)

net = Net_3DFR(img_h, img_w, img_c)
net.build(1, 100, loaders=loaders, train=False)
net.load_model(dir_model)
net.test(loaders, dir_output)
