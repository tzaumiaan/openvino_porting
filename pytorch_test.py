import torch
from PIL import Image
import numpy as np
import os
from datetime import datetime

from model_utils import count_param_size, count_flops, get_model, get_trans

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
  logits = logits_tensor.cpu().detach().numpy().squeeze()
  return logits

def print_logits(logits, model_name, framework_name, k=5):
  from imagenet_labels import labelmap
  softmax = np.exp(logits)/np.sum(np.exp(logits))
  indices = logits.argsort()[::-1][:k]
  print('{} inference result: {}'.format(framework_name, model_name))
  for i in indices:
    print('{:8.5f}({:5.3f}): {}'.format(logits[i], softmax[i], labelmap[i]))

def pytorch_speed_test(model, target_size):
  from imagenet_labels import labelmap
  if torch.cuda.is_available():
    model = model.cuda()
  #model = model.cpu()
  # inference speed test
  n_trials = 1000
  n_warmup = 100
  exec_time = 0
  for i in range(n_trials):
    image_tensor = torch.rand(1, 3, target_size, target_size)
    if torch.cuda.is_available():
      image_tensor = image_tensor.cuda()
    t0 = datetime.now()
    logits_tensor = model(image_tensor)
    t1 = datetime.now()
    dt = (t1-t0).total_seconds()
    logits = logits_tensor.cpu().detach().numpy().squeeze()
    top1 = logits.argsort()[::-1][0]
    if i >= n_warmup:
        exec_time += dt
    if i%100 == 0:
        print('trial {} takes {:0.4f}secs: logit={:.5f} label={}'.format(i, dt, logits[top1], labelmap[top1]))

  print('average inference time {:.1f}ms'.format(exec_time/(n_trials-n_warmup)*1e3))

# main entry
if __name__=='__main__':
  import argparse 
  parser = argparse.ArgumentParser(description='model conversion test')
  parser.add_argument('--model_name', type=str, default='resnet50')
  parser.add_argument('--image_file', type=str, default='dog-komondor.jpg')
  args = parser.parse_args()

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
  print('==== pytorch speed test ====')
  pytorch_speed_test(model, target_size)
