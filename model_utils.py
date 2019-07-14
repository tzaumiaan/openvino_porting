import numpy as np
from torchvision import models, transforms
import torch

def count_param_size(model):
  # print parameter size of model
  n_params = np.sum([np.prod(p.size()) for p in model.parameters() if p.requires_grad])
  return n_params

def count_flops(model, target_size):
  from thop import profile
  model_input = torch.randn(1, 3, target_size, target_size)
  flops, n_params = profile(model, inputs=(model_input, ), verbose=False)
  return flops, n_params

# model instance
def get_model(model_name):
  if model_name == 'inception_v3':
    model = models.inception_v3(pretrained=True, transform_input=False)
  elif model_name == 'inception_v4':
    from model_zoo.inceptionv4 import inceptionv4
    model = inceptionv4(pretrained='imagenet')
  elif model_name == 'inception_resnet_v2':
    from model_zoo.inceptionresnetv2 import inceptionresnetv2
    model = inceptionresnetv2(pretrained='imagenet')
  elif model_name == 'xception':
    from model_zoo.xception import xception
    model = xception(pretrained='imagenet')
  elif model_name == 'resnet50':
    model = models.resnet50(pretrained=True)
  elif model_name == 'resnet101':
    model = models.resnet101(pretrained=True)
  elif model_name == 'resnet152':
    model = models.resnet152(pretrained=True)
  elif model_name == 'resnext50_32x4d':
    model = models.resnext50_32x4d(pretrained=True)
  elif model_name == 'resnext101_32x8d':
    model = models.resnext101_32x8d(pretrained=True)
  elif model_name == 'densenet121':
    model = models.densenet121(pretrained=True)
  elif model_name == 'densenet169':
    model = models.densenet169(pretrained=True)
  elif model_name == 'densenet201':
    model = models.densenet201(pretrained=True)
  elif model_name == 'densenet161':
    model = models.densenet161(pretrained=True)
  elif model_name == 'squeezenet1_0':
    model = models.squeezenet1_0(pretrained=True)
  elif model_name == 'squeezenet1_1':
    model = models.squeezenet1_1(pretrained=True)
  elif model_name == 'mobilenet_v2':
    from model_zoo.mobilenet import mobilenet_v2
    model = mobilenet_v2(pretrained=True)
  elif model_name == 'shufflenet_v2_x1_0':
    from model_zoo.shufflenetv2 import shufflenet_v2_x1_0
    model = shufflenet_v2_x1_0(pretrained=True)
  elif model_name == 'shufflenet_v2_x0_5':
    from model_zoo.shufflenetv2 import shufflenet_v2_x0_5
    model = shufflenet_v2_x0_5(pretrained=True)
  elif model_name == 'nasnet_a_large':
    # Pad layer not supported by OpenVINO
    from model_zoo.nasnet import nasnetalarge
    model = nasnetalarge(num_classes=1000, pretrained='imagenet')
  elif model_name == 'nasnet_a_mobile':
    # Pad layer not supported by OpenVINO
    from model_zoo.nasnet_mobile import nasnetamobile
    model = nasnetamobile(num_classes=1000, pretrained='imagenet')
  elif model_name == 'efficientnet_b0':
    from model_zoo.efficientnet_pytorch import EfficientNet
    model = EfficientNet.from_pretrained('efficientnet-b0')
  elif model_name == 'efficientnet_b1':
    from model_zoo.efficientnet_pytorch import EfficientNet
    model = EfficientNet.from_pretrained('efficientnet-b1')
  elif model_name == 'efficientnet_b2':
    from model_zoo.efficientnet_pytorch import EfficientNet
    model = EfficientNet.from_pretrained('efficientnet-b2')
  elif model_name == 'efficientnet_b3':
    from model_zoo.efficientnet_pytorch import EfficientNet
    model = EfficientNet.from_pretrained('efficientnet-b3')
  elif model_name == 'efficientnet_b4':
    from model_zoo.efficientnet_pytorch import EfficientNet
    model = EfficientNet.from_pretrained('efficientnet-b4')
  elif model_name == 'efficientnet_b5':
    from model_zoo.efficientnet_pytorch import EfficientNet
    model = EfficientNet.from_pretrained('efficientnet-b5')
  else:
    raise ValueError('invalid model')
  
  # evaluation mode
  model = model.eval()
  
  return model

# transformation
def get_trans(model_name):
  # norm_mean and norm_std defined as RGB
  imagenet_mean = [0.485, 0.456, 0.406]
  imagenet_std = [0.229, 0.224, 0.225]
  tf_slim_mean = [0.5, 0.5, 0.5]
  tf_slim_std = [0.5, 0.5, 0.5]
  if model_name.startswith('inception') or model_name.startswith('xception'):
    target_size = 299
    norm_mean = tf_slim_mean
    norm_std = tf_slim_std
  elif model_name == 'nasnet_a_large':
    target_size = 331
    norm_mean = tf_slim_mean
    norm_std = tf_slim_std
  elif model_name == 'nasnet_a_mobile':
    target_size = 224
    norm_mean = tf_slim_mean
    norm_std = tf_slim_std
    norm_mean = imagenet_mean
    norm_std = imagenet_std
  elif model_name.startswith('efficientnet'):
    if model_name.endswith('0'): target_size = 224
    elif model_name.endswith('1'): target_size = 240
    elif model_name.endswith('2'): target_size = 260
    elif model_name.endswith('3'): target_size = 300
    elif model_name.endswith('4'): target_size = 380
    elif model_name.endswith('5'): target_size = 456
    norm_mean = imagenet_mean
    norm_std = imagenet_std
  else:
    target_size = 224
    norm_mean = imagenet_mean
    norm_std = imagenet_std
  print('target size = {}'.format(target_size))
  print('normalization mean = {}, std = {}'.format(norm_mean, norm_std))
    
  # transformation
  trans = transforms.Compose([
    transforms.Resize(int(target_size/224*256)),
    transforms.CenterCrop(target_size),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std)
  ])

  return target_size, trans
