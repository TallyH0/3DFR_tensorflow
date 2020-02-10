from model import Net_3DFR
from data_loader import Data_loader_sequence

img_h, img_w, img_c = 240, 360, 1
batch_size = 1
max_epoch = 100

loaders = []

dir_datas = [
  'F:/Data/CDNet2014/dataset/baseline/highway',
  'F:/Data/CDNet2014/dataset/baseline/office',
  'F:/Data/CDNet2014/dataset/baseline/PETS2006'
]
dir_model = 'model_CDNet2014_baseline'

for dir_data in dir_datas:
  loaders.append(Data_loader_sequence(dir_data, img_h, img_w))


net = Net_3DFR(img_h, img_w, img_c)
net.build(batch_size, max_epoch, loaders=loaders)
net.train(dir_model)
