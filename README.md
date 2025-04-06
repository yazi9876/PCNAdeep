bin/ 文件夹这个一般是 脚本入口或测试文件的放置处，可能你现在还没看到内容，暂时不用管
d2_mod/ 文件夹（= Detectron2 的修改版）这个文件夹是重点！是对原生 Detectron2 模型做了一些定制修改
dataset_mapper.py图像读取（包括 PCNA 和明场图合成）图像预处理（gamma 校正、增强）把 dataset_dict 转换成 Tensor ✅ 是整个训练时的“图像准备入口”
fast_rcnn.py ✅这个是对原始 Detectron2 的 fast_rcnn_inference_single_image() 做了修改，用于：在预测时输出 confidence score控制哪些 box 被筛选出来
pcnaDeep/ 文件夹 ✅这是主代码文件夹，包含了真正执行“检测+追踪+细胞周期判断”的逻辑
detect.py ✅负责模型预测阶段，调用 Detectron2 模型进行：图像分割 / 检测输出 mask、bbox、score如果加上 --vis_out 参数，会保存可视化图像
main.py ✅你命令行运行的主要脚本（主程序入口）train_detectron2.py ✅这是你训练模型的脚本：这是你训练模型的脚本：加载配置（dtrnCfg.yaml）注册数据集（Detectron2 dataset）调用训练器开始训练！
config/ 文件夹用来存放 yaml 配置文件，也就是控制模型结构、数据路径、训练参数的地方。

deprecated/（弃用的旧代码
deep_cell_functions/（PCNA识别工具函数）
detectron2-0.4_mod/就是 pcnaDeep 所用的 修改版本的 Detectron2
docs/一般是文档，生成网页用的（你可以忽略）
examples/一些示例图像、测试用的小文件夹，可以参考用法，但不是 pipeline 核心
models/这个一般是用来存放：下载的模型权重文件或训练后输出的模型文件。如果你训练自己的模型，最终的 .pth 就会保存在这里

tutorial/assets/可能包含测试图像、标签或演示用的小模型，用于 notebook 示例运行。
------------------------------------------------------------------------------------------------------------------------------------

# pcnaDeep: a deep-learning based single-cell cycle profiler with PCNA signal

Welcome! pcnaDeep integrates cutting-edge detection techniques with tracking and cell cycle resolving models.
Using the Mask R-CNN model under FAIR's Detectron2 framework, pcnaDeep is able to detect and resolve very dense cell tracks with __PCNA fluorescence__.

<img src="/tutorial/assets/overview.jpg" alt="overview" width="800" />

## Installation
1. PyTorch (torch >= 1.7.1) installation and CUDA GPU support are essential. Visit [PyTorch homepage](https://pytorch.org/) for specific installation schedule.

- Check the GPU and PyTorch are available:
   ```
   import torch
   print(torch.cuda.is_available())
   ```

2. Install modified __Detectron2 v0.4__ in this directory ([original package homepage](https://github.com/facebookresearch/detectron2))

   ```angular2html
      cd detectron2-04_mod
      pip install .
   ```

   <details>
   <summary>Building detectron2 on Windows? Click here.
   </summary>

      - Before building detectron2, you must install <a title="Microsoft Visual C++" href="https://visualstudio.microsoft.com/vs/features/cplusplus/">Microsoft Visual C++</a> (please use the standard installation).
      After installation, please restart your system.
      - If your torch version is old, the following changes of the `torch` package may be required. <a title="Ref" href="https://blog.csdn.net/weixin_42644340/article/details/109178660">Reference (Chinese)</a>.

         ```angular2html
            In torch\include\torch\csrc\jit\argument_spec.h,
            static constexpr size_t DEPTH_LIMIT = 128;
               change to -->
            static const size_t DEPTH_LIMIT = 128;
         ```
   </details>

   ---

   In pcnaDeep, the detectron2 v0.4 dependency has been modified in two ways:
      1. To generate confidence score output of the instance classification, the method `detectron2.modeling.roi_heads.fast_rcnn.fast_rcnn_inference_single_image` has been modified.
      2. A customized dataset mapper function has been implemented as `detectron2.data.dataset_mapper.read_PCNA_training`.


3. Install pcnaDeep from source in this directory
   ```
   cd bin
   python setup.py install
   ```
4. (optional, for training data annotation only) Download [VGG Image Annotator 2](https://www.robots.ox.ac.uk/~vgg/software/via/) software.
5. (optional, for visualisation only) Install [Fiji (ImageJ)](https://fiji.sc/) with [TrackMate CSV Importer](https://github.com/tinevez/TrackMate-CSVImporter) plugin.


## Demo data download

All demo data are stored at [Zenodo](https://zenodo.org/record/5515771#.YqAISRNBxxg).

### Download pre-trained Mask R-CNN weights

The Mask R-CNN is trained on 60X microscopic images sized 1200X1200 square pixels. [Download here](https://zenodo.org/record/5515771/files/mrcnn_sat_rot_aug.pth?download=1).

You must download pre-trained weights and save it under `~/models/` for running tutorials.

### Download example datasets

You may need to download [some example datasets](https://github.com/chan-labsite/PCNAdeep/tree/main/examples) to run tutorials (like the quick-start guide below).

## Getting started

See [a quick tutorial](tutorial/getting_started.ipynb) to get familiar with pcnaDeep.

You may also go through other tutorials for advanced usages.

## API Documentation

API documentation is available [here](https://pcnadeep.readthedocs.io/en/latest/index.html).

## Reference

Please cite our paper if you found this package useful. 
```
pcnaDeep: A Fast and Robust Single-Cell Tracking Method Using Deep-Learning Mediated Cell Cycle Profiling
Yifan Gui, Shuangshuang Xie, Yanan Wang, Ping Wang, Renzhi Yao, Xukai Gao, Yutian Dong, Gaoang Wang, Kuan Yoow Chan
Bioinformatics, Volume 38, Issue 20, 15 October 2022, Pages 4846–4847; doi: https://doi.org/10.1093/bioinformatics/btac602
```

## Licence

pcnaDeep is released under the [Apache 2.0 license](LICENSE).
