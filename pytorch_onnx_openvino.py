import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
import os
import onnx
import onnxruntime
from openvino.inference_engine import IENetwork, IEPlugin
from datetime import datetime

# model instance
def get_model(model_name):
  if model_name == 'inception_v3':
    model = models.inception_v3(pretrained=True, transform_input=False)
  elif model_name == 'inception_v4':
    from model_zoo.inceptionv4 import inceptionv4
    model = inceptionv4(pretrained='imagenet')
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
  elif model_name == 'efficientnet_b4':
    from model_zoo.efficientnet_pytorch import EfficientNet
    model = EfficientNet.from_pretrained('efficientnet-b4')
  else:
    raise ValueError('invalid model')
  
  # evaluation mode
  model = model.eval()
  
  return model

def print_param_size(model_name, model):
  # print parameter size of model
  n_params = np.sum([np.prod(p.size()) for p in model.parameters() if p.requires_grad])
  print('{} has {}M trainable parameters'.format(model_name, n_params/1e6))

def get_trans(model_name):
  # norm_mean and norm_std defined as RGB
  imagenet_mean = [0.485, 0.456, 0.406]
  imagenet_std = [0.229, 0.224, 0.225]
  tf_slim_mean = [0.5, 0.5, 0.5]
  tf_slim_std = [0.5, 0.5, 0.5]
  if model_name in ['inception_v3', 'inception_v4', 'xception']:
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
  elif model_name.startswith('efficientnet'):
    target_size = 240
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

def run_inference_pytorch(image_file, model_name, model, trans):
  from imagenet_labels import labelmap
  image = Image.open(image_file)
  image_tensor = trans(image)
  
  # extend batch dimension
  image_tensor = image_tensor.unsqueeze(dim=0)
  
  # apply cuda
  if torch.cuda.is_available():
    model = model.cuda()
    image_tensor = image_tensor.cuda()

  logits = model(image_tensor)

  k = 5
  values, indices = torch.topk(logits, k)
  print('pytorch inference result: {}'.format(model_name))
  for i in range(k):
    print(values[0][i].item(), labelmap[indices[0][i].item()])

def image_to_npz(image_file, trans, npz_file):
  image = Image.open(image_file)
  image_tensor = trans(image).unsqueeze(dim=0)
  print('image tensor shape = ', image_tensor.shape)
  print('top left patch (green channel) = ', image_tensor[0,1,:2,:2])
  np.savez(npz_file, image_tensor)

  # test if it can be loaded correctly
  image_file = np.load(npz_file)
  image_np = image_file['arr_0']
  image_file.close()
  print('image numpy array shape = ', image_np.shape)
  print('top left patch (green channel) = ', image_np[0,1,:2,:2])

def convert_pytorch_to_onnx(model_name, model, target_size):
  onnx_file = 'model_{}.onnx'.format(model_name)
  input_tensor = torch.ones([1, 3, target_size, target_size], dtype=torch.float32)
  # export the model
  torch.onnx.export(model.cpu(), input_tensor, onnx_file, export_params=True, verbose=False)
  # optimization
  model_onnx = onnx.load(onnx_file)
  from onnx import optimizer
  passes = ["extract_constant_to_initializer", "eliminate_unused_initializer"]
  model_onnx_opt = optimizer.optimize(model_onnx, passes)
  onnx.save(model_onnx_opt, onnx_file)
  # print model information
  model_onnx = onnx.load(onnx_file)
  onnx.checker.check_model(model_onnx) # will print nothing if nothing wrong
  
def run_inference_onnx(model_name, npz_file):
  from imagenet_labels import labelmap
  # input np array
  image_file = np.load(npz_file)
  image_np = image_file['arr_0']
  # model object
  onnx_file = 'model_{}.onnx'.format(model_name)
  onnx_session = onnxruntime.InferenceSession(onnx_file)
  input_blob = onnx_session.get_inputs()[0].name
  output_blob = onnx_session.get_outputs()[0].name
  # run model
  res = onnx_session.run([output_blob], {input_blob: image_np})
  logits = res[0].squeeze()
  # print result
  k = 5
  indices = logits.argsort()[::-1][:k]
  print('onnx inference result: {}'.format(model_name))
  for i in indices:
    print(logits[i], labelmap[i])

def convert_onnx_to_openvino(model_name):
  import os
  onnx_file = 'model_{}.onnx'.format(model_name)
  mo_py = '/opt/intel/openvino/deployment_tools/model_optimizer/mo.py'
  os.system('python {} --input_model {}'.format(mo_py, onnx_file))

def load_openvino_model(device, model_xml, model_bin):
  plugin = IEPlugin(device=device, plugin_dirs=None)
  net = IENetwork(model=model_xml, weights=model_bin)
  supported_layers = plugin.get_supported_layers(net)
  not_supported_layers = [l for l in net.layers.keys() if l not in supported_layers]
  if len(not_supported_layers) != 0:
    print("Following layers are not supported by the plugin for specified device {}:\n {}".format(
        plugin.device, ', '.join(not_supported_layers)))
  else:
    print("All layers supported")
  exec_net = plugin.load(network=net)
  input_blob = next(iter(net.inputs))
  output_blob = next(iter(net.outputs))
  return plugin, exec_net, input_blob, output_blob

def run_inference_openvino(model_name, npz_file):
  from imagenet_labels import labelmap
  # input np array
  image_file = np.load(npz_file)
  image_np = image_file['arr_0']
  # load model
  xml_file = 'model_{}.xml'.format(model_name)
  bin_file = 'model_{}.bin'.format(model_name)
  plugin, exec_net, input_blob, output_blob = load_openvino_model('CPU', xml_file, bin_file)
  # run inference
  res = exec_net.infer(inputs={input_blob: image_np})
  logits = res[output_blob].squeeze()
  # print result
  k = 5
  indices = logits.argsort()[::-1][:k]
  print('openvino inference result: {}'.format(model_name))
  for i in indices:
    print(logits[i], labelmap[i])
  # clean up
  del plugin, exec_net

def openvino_speed_test(model_name):
  from imagenet_labels import labelmap
  
  # load model
  xml_file = 'model_{}.xml'.format(model_name)
  bin_file = 'model_{}.bin'.format(model_name)
  plugin, exec_net, input_blob, output_blob = load_openvino_model('CPU', xml_file, bin_file)

  # inference speed test
  n_trials = 1000
  n_warmup = 100
  exec_time = 0
  for i in range(n_trials):
    random_input = np.random.randn(3, target_size, target_size)
    t0 = datetime.now()
    res = exec_net.infer(inputs={input_blob: random_input})
    t1 = datetime.now()
    dt = (t1-t0).total_seconds()
    logits = res[output_blob].squeeze()
    top1 = logits.argsort()[::-1][0]
    if i >= n_warmup:
        exec_time += dt
    if i%100 == 0:
        print('trial {} takes {:0.4f}secs: logit={} label={}'.format(i, dt, logits[top1], labelmap[top1]))

  print('average inference time {:0.4f}secs'.format(exec_time/(n_trials-n_warmup)))
  
  # clean up
  del plugin, exec_net

# main entry
if __name__=='__main__':
  import argparse 
  parser = argparse.ArgumentParser(description='model conversion test')
  parser.add_argument('--model_name', type=str, default='resnet50')
  parser.add_argument('--image_file', type=str, default='dog-komondor.jpg')
  args = parser.parse_args()

  npz_file = 'test_input.npz'
  
  # pytorch
  print('==== pytorch ====')
  model = get_model(args.model_name)
  print_param_size(args.model_name, model)
  target_size, trans = get_trans(args.model_name)
  run_inference_pytorch(args.image_file, args.model_name, model, trans)
  
  # onnx
  print('==== onnx ====')
  image_to_npz(args.image_file, trans, npz_file)
  convert_pytorch_to_onnx(args.model_name, model, target_size)
  run_inference_onnx(args.model_name, npz_file)

  # openvino
  print('==== openvino ====')
  convert_onnx_to_openvino(args.model_name)
  run_inference_openvino(args.model_name, npz_file)
  print('==== openvino speed test ====')
  openvino_speed_test(args.model_name)

