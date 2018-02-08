# PyTorch implementation of the [YOLO (You Only Look Once) v2](https://arxiv.org/pdf/1612.08242.pdf)

The YOLOv2 is one of the most popular [one-stage](https://arxiv.org/abs/1708.02002) object detector.
This project adopts [PyTorch](http://pytorch.org/) as the developing framework to increase productivity, and utilize [ONNX](https://github.com/onnx/onnx) to convert models into [Caffe 2](https://caffe2.ai/) to benefit engineering deployment.
If you are benefited from this project, a donation will be appreciated (via [PayPal](https://www.paypal.me/minimumshen), [微信支付](donate_mm.jpg) or [支付宝](donate_alipay.jpg)).

![](demo.gif)

## Designs

- Flexible configuration design.
Program settings are configurable and can be modified (via **configure file overlaping** (-c/--config option) or **command editing** (-m/--modify option)) using command line argument.

- Monitoring via [TensorBoard](https://github.com/tensorflow/tensorboard).
Such as the loss values and the debugging images (such as IoU heatmap, ground truth and predict bounding boxes).

- Parallel model training design.
Different models are saved into different directories so that can be trained simultaneously.

- Using a NoSQL database to store evaluation results with multiple dimension of information.
This design is useful when analyzing a large amount of experiment results.

- Time-based output design.
Running information (such as the model, the summaries (produced by TensorBoard), and the evaluation results) are saved periodically via a predefined time.

- Checkpoint management.
Several latest checkpoint files (.pth) are preserved in the model directory and the older ones are deleted.

- NaN debug.
When a NaN loss is detected, the running environment (data batch) and the model will be exported to analyze the reason.

- Unified data cache design.
Various dataset are converted into a unified data cache via corresponding cache plugins.
Some plugins are already implemented. Such as [PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/) and [MS COCO](http://cocodataset.org/).

- Arbitrarily replaceable model plugin design.
The main deep neural network (DNN) can be easily replaced via configuration settings.
Multiple models are already provided. Such as Darknet, [ResNet](https://arxiv.org/abs/1512.03385), Inception [v3](https://arxiv.org/abs/1512.00567) and [v4](https://arxiv.org/abs/1602.07261), [MobileNet](https://arxiv.org/abs/1704.04861) and [DenseNet](https://arxiv.org/abs/1608.06993).

- Extendable data preprocess plugin design.
The original images (in different sizes) and labels are processed via a sequence of operations to form a training batch (images with the same size, and bounding boxes list are padded).
Multiple preprocess plugins are already implemented. Such as
augmentation operators to process images and labels (such as random rotate and random flip) simultaneously,
operators to resize both images and labels into a fixed size in a batch (such as random crop),
and operators to augment images without labels (such as random blur, random saturation and random brightness).

## Feautures

- [x] Reproduce the original paper's training results.
- [x] Multi-scale training.
- [x] Dimension cluster.
- [x] [Darknet](http://pjreddie.com) model file (`.weights`) parser.
- [x] Detection from image and camera.
- [x] Processing Video file.
- [x] Multi-GPU supporting.
- [ ] Distributed training.
- [ ] [Focal loss](https://arxiv.org/abs/1708.02002).
- [x] Channel-wise model parameter analyzer.
- [x] Automatically change the number of channels.
- [x] Receptive field analyzer.

## Quick Start

This project uses [Python 3](https://www.python.org/). To install the dependent libraries, type the following command in a terminal.

```
sudo pip3 install -r requirements.txt
```

`quick_start.sh` contains the examples to perform detection and evaluation. Run this script.
Multiple datasets and models (the original Darknet's format, will be converted into PyTorch's format) will be downloaded ([aria2](https://aria2.github.io/) is required).
These datasets are cached into different data profiles, and the models are evaluated over the cached data.
The models are used to detect objects in an example image, and the detection results will be shown.

## License

This project is released as the open source software with the GNU Lesser General Public License version 3 ([LGPL v3](http://www.gnu.org/licenses/lgpl-3.0.html)).
