import numpy as np
import os
from openvino.inference_engine import IENetwork, IEPlugin
from datetime import datetime

OUTPUT_DIR = 'model_converted'


def print_logits(logits, model_name, framework_name, k=5):
  from imagenet_labels import labelmap
  softmax = np.exp(logits)/np.sum(np.exp(logits))
  indices = logits.argsort()[::-1][:k]
  print('{} inference result: {}'.format(framework_name, model_name))
  for i in indices:
    print('{:8.5f}({:5.3f}): {}'.format(logits[i], softmax[i], labelmap[i]))


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

def print_perf_counts(exec_net):
  perf_counts = exec_net.requests[0].get_perf_counts()
  print('OpenVINO performance report')
  #print("{:<70} {:<15} {:<15} {:<15} {:<10}".format('name', 'layer_type', 'exet_type', 'status', 'real_time, us'))
  layer_dict = {}
  for layer, stats in perf_counts.items():
    #print("{:<70} {:<15} {:<15} {:<15} {:<10}".format(
    #    layer, stats['layer_type'], stats['exec_type'], stats['status'], stats['real_time']))
    if stats['layer_type'] in layer_dict:
      layer_dict[stats['layer_type']] += int(stats['real_time'])
    else:
      layer_dict[stats['layer_type']] = int(stats['real_time'])
  layer_types, layer_times = list(layer_dict.keys()), list(layer_dict.values())
  layer_time_ratio = layer_times / np.sum(layer_times)
  ranking = layer_time_ratio.argsort()[::-1]
  print('Run time ranking by layer types')
  print('{:<15} {:<10} {:<10}'.format('layer_type', 'real_time', 'percentage'))
  for i in ranking:
    print('{:<15} {:>10} {:>10.6f}%'.format(layer_types[i], layer_times[i], layer_time_ratio[i]*100))
  print('IE inference time {:0.3f}ms'.format(np.sum(layer_times)*1e-3))

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
    random_input = np.random.randn(3, 224, 224)
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

  print('average inference time {:.1f}ms'.format(exec_time/(n_trials-n_warmup)*1e3))
  
  print_perf_counts(exec_net)
  
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
  

  # openvino
  print('==== openvino ====')
  logits_openvino = run_inference_openvino(args.model_name, npz_file)
  print_logits(logits_openvino, args.model_name, 'openvino')
  print('==== openvino speed test ====')
  openvino_speed_test(args.model_name)
