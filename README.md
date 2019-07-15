# openvino_porting
Porting pretrained models to OpenVINO

## Environment setup
Install OpenVINO following the instructions:
https://software.intel.com/en-us/openvino-toolkit

Setup a virtual environment with `python3` and its `pip` module,
or with `conda` (from Miniconda or Anaconda).
Then install packages through the specific package list based on
the configuration of the machine. 

This is the `virtuelenv` with `pip` example.
```
$ mkvirtualenv env -p python3
(env)$ pip install -r package_list_pip.txt
```

Or using `conda`, where the following is an example of environment
set with CUDA 10.
```
$ conda env create -f package_list_conda_py37_cu10.yml
$ conda activate py37-torch110-cu10
```

## Try model conversion
Specify a model name (`inception_v3` as example here) and run the line.
```
(env)$ python pytorch_onnx_openvino.py --model_name inception_v3
```

## Credits
Great appreciation towards the following GitHub repos who really enable
the possibilities to easily use the pretrained models.
- @Cadene for https://github.com/Cadene/pretrained-models.pytorch
- @lukemelas for https://github.com/lukemelas/EfficientNet-PyTorch

Great appreciation towards the THOP tool developed by:
- @Lyken17 for https://github.com/Lyken17/pytorch-OpCounter
