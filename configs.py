import json
import configparser

def pasre_config(path_config):
  config = configparser.ConfigParser()
  config.read(path_config)
  config_dict = config.defaults()

  config_dict['dir_data'] = json.loads(config_dict['dir_data'].replace('\'', '\"'))
  config_dict['img_h'] = int(config_dict['img_h'])
  config_dict['img_w'] = int(config_dict['img_w'])
  config_dict['batch_size'] = int(config_dict['batch_size'])
  config_dict['max_epoch'] = int(config_dict['max_epoch'])
  config_dict['gray'] = bool(config_dict['gray'])

  return config_dict

if __name__ == '__main__':
  configs = pasre_config('config_cameraJitter.ini')
  print(configs)