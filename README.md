# openvino_porting
Porting pretrained models to OpenVINO

## Environment setup
Install OpenVINO following the instructions:
https://software.intel.com/en-us/openvino-toolkit

Setup a virtual environment with `python3` and its `pip` module.
Then install packages through the specific package list based on
the configuration of the machine. 
The following example assumes no CUDA GPU is available.
```
(venv)$ pip install -r package_list_cpu.txt
```

## Try model conversion
Specify a model name (`inception_v3` as example here) and run the line.
```
(venv)$ python pytorch_onnx_openvino.py --model_name inception_v3
```

## Credits
Great appreciation towards the following GitHub repos who really enable
the possibilities to easily use the pretrained models.
- @Cadene for https://github.com/Cadene/pretrained-models.pytorch
- @lukemelas for https://github.com/lukemelas/EfficientNet-PyTorch

Great appreciation towards the THOP tool developed by:
- @Lyken17 for https://github.com/Lyken17/pytorch-OpCounter
