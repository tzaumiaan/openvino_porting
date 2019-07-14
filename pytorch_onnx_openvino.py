import torch
from PIL import Image
import numpy as np
import os
import onnx
import onnxruntime
from openvino.inference_engine import IENetwork, IEPlugin
from datetime import datetime
import os

from model_utils import count_param_size, count_flops, get_model, get_trans

OUTPUT_DIR = 'model_converted'

def run_inference_pytorch(image_file, model, trans):
  image = Image.open(image_file)
  image_tensor = trans(image)
  
  # extend batch dimension
  image_tensor = image_tensor.unsqueeze(dim=0)
  
  # apply cuda
  if torch.cuda.is_available():
    model = model.cuda()
    image_tensor = image_tensor.cuda()

  logits_tensor = model(image_tensor)
  logits = logits_tensor.detach().numpy().squeeze()
  return logits

def print_logits(logits, model_name, framework_name, k=5):
  from imagenet_labels import labelmap
  softmax = np.exp(logits)/np.sum(np.exp(logits))
  indices = logits.argsort()[::-1][:k]
  print('{} inference result: {}'.format(framework_name, model_name))
  for i in indices:
    print('{:8.5f}({:5.3f}): {}'.format(logits[i], softmax[i], labelmap[i]))

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
  onnx_file = os.path.join(OUTPUT_DIR, 'model_{}.onnx'.format(model_name))
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
  # input np array
  image_file = np.load(npz_file)
  image_np = image_file['arr_0']
  # model object
  onnx_file = os.path.join(OUTPUT_DIR, 'model_{}.onnx'.format(model_name))
  onnx_session = onnxruntime.InferenceSession(onnx_file)
  input_blob = onnx_session.get_inputs()[0].name
  output_blob = onnx_session.get_outputs()[0].name
  # run model
  res = onnx_session.run([output_blob], {input_blob: image_np})
  logits = res[0].squeeze()
  return logits

def convert_onnx_to_openvino(model_name):
  onnx_file = 'model_{}.onnx'.format(model_name)
  mo_py = '/opt/intel/openvino/deployment_tools/model_optimizer/mo.py'
  return_path = os.getcwd()
  os.chdir(OUTPUT_DIR)
  os.system('python {} --input_model {}'.format(mo_py, onnx_file))
  os.chdir(return_path)

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
  # input np array
  image_file = np.load(npz_file)
  image_np = image_file['arr_0']
  # load model
  xml_file = os.path.join(OUTPUT_DIR, 'model_{}.xml'.format(model_name))
  bin_file = os.path.join(OUTPUT_DIR, 'model_{}.bin'.format(model_name))
  plugin, exec_net, input_blob, output_blob = load_openvino_model('CPU', xml_file, bin_file)
  # run inference
  res = exec_net.infer(inputs={input_blob: image_np})
  logits = res[output_blob].squeeze()
  # clean up
  del plugin, exec_net
  return logits

def openvino_speed_test(model_name):
  from imagenet_labels import labelmap
  
  # load model
  xml_file = os.path.join(OUTPUT_DIR, 'model_{}.xml'.format(model_name))
  bin_file = os.path.join(OUTPUT_DIR, 'model_{}.bin'.format(model_name))
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
        print('trial {} takes {:0.4f}secs: logit={:.5f} label={}'.format(i, dt, logits[top1], labelmap[top1]))

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
  
  if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

  # pytorch
  print('==== pytorch ====')
  model = get_model(args.model_name)
  n_params = count_param_size(model)
  print('{} has {:.4f}M trainable parameters'.format(args.model_name, n_params/1e6))
  target_size, trans = get_trans(args.model_name)
  flops, n_params = count_flops(model, target_size)
  print('THOP: {} has {:.4f}M trainable parameters'.format(args.model_name, n_params/1e6))
  print('THOP: {} has {:.4f}G FLOPs'.format(args.model_name, flops/1e9))
  logits_pytorch = run_inference_pytorch(args.image_file, model, trans)
  print_logits(logits_pytorch, args.model_name, 'pytorch')
  
  # onnx
  print('==== onnx ====')
  image_to_npz(args.image_file, trans, npz_file)
  convert_pytorch_to_onnx(args.model_name, model, target_size)
  logits_onnx = run_inference_onnx(args.model_name, npz_file)
  print_logits(logits_onnx, args.model_name, 'onnx')
  print('Logits avg diff: {}'.format(np.mean(logits_onnx - logits_pytorch)))

  # openvino
  print('==== openvino ====')
  convert_onnx_to_openvino(args.model_name)
  logits_openvino = run_inference_openvino(args.model_name, npz_file)
  print_logits(logits_openvino, args.model_name, 'openvino')
  print('Logits avg diff: {}'.format(np.mean(logits_openvino - logits_pytorch)))
  print('==== openvino speed test ====')
  openvino_speed_test(args.model_name)
