

# Action recognition using 3D ResNet: 
Forked from https://github.com/kenshohara/video-classification-3d-cnn-pytorch with the 
purpose of providing more detailed instructions and some scripts in order to make the analysis easier. 

More details about 3D ResNet can be found here (ResNeXt-101 using the cardinality
of 32): https://arxiv.org/pdf/1711.09577.pdf

Tested with Python3 on Google Compute Engine using the following configuration: 

```
OS: Linux Debian 9
Machine type: n1-highmem-2 (2 vCPUs, 13 GB memory)
CPU platform: Intel Sandy Bridge
GPUs: 1 x NVIDIA Tesla K80
Zone: europe-west1-b
```

Follow the steps below to install and run this code:

## 1. Install prerequisites (GCE Linux Debian 9): 


```
sudo apt install bzip2, gcc, make, cmake, linux-source, linux-headers-`uname -r`, git
```

## 2. Install Anaconda
```
wget https://repo.anaconda.com/archive/Anaconda3-5.1.0-Linux-x86_64.sh
chmod +x Anaconda3-5.1.0-Linux-x86_64.sh
./Anaconda3-5.1.0-Linux-x86_64.sh
source .bashrc
```

## 3. Get [Nvidia driver](http://www.nvidia.com/download/driverResults.aspx/122818/en-us):

```
wget http://us.download.nvidia.com/XFree86/Linux-x86_64/384.66/NVIDIA-Linux-x86_64-384.66.run
chmod +x NVIDIA-Linux-x86_64-340.65.run
sudo ./NVIDIA-Linux-x86_64-340.65.run
```

## 4. Install PyTorch and CUDA

```
conda install pytorch torchvision cuda80 -c soumith
```

## 5. Install FFmpeg, FFprobe

```
wget http://johnvansickle.com/ffmpeg/releases/ffmpeg-release-64bit-static.tar.xz
tar xvf ffmpeg-release-64bit-static.tar.xz
cd ./ffmpeg-3.3.3-64bit-static/; sudo cp ffmpeg ffprobe /usr/local/bin;
```

## 6. GET this repository

```
git clone https://github.com/rlica/video-classification-3d-cnn-pytorch.git
```

## 7. Download pretrained model from Google Drive:

Add this to the .bashrc file in order to download files from Google Drive:

```
function gdrive_download () {
  CONFIRM=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://docs.google.com/uc?export=download&id=$1" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')
  wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$CONFIRM&id=$1" -O $2
  rm -rf /tmp/cookies.txt
}
source .bashrc
```

Then download any [pretrained model](https://drive.google.com/drive/folders/14KRBqT8ySfPtFSuLsFS2U4I-ihTDs0Y9?usp=sharing) using the file IDs:

```
mkdir models && cd models
gdrive_download 1NOSyEnwPuEdtWHnsVC8JrDfL0sY3iKZV resnext-101-kinetics.pth
```

## 8. Download some video files (eg: [THUMOS14](http://crcv.ucf.edu/THUMOS14/test_set/TH14_test_set_mp4/) ):

```
mkdir videos && cd videos
wget http://crcv.ucf.edu/THUMOS14/test_set/TH14_test_set_mp4/video_test_0000002.mp4
```

## 9. Run the CODE:


### 9.1 Process files


Assume input video files are located in ./videos and the 'input' file contains the names of the video files there.
To calculate class scores for each 16 frames, use --mode score.

```
python main.py --input ./input --video_root ./videos --output ./output.json --model ./resnet-34-kinetics.pth --mode score
```
This command can be found in the  ```./run.sh ``` file

### 9.2 Result Video Generation
This is a Python script for generating videos of classification results.  
It uses both ```output.json``` and videos as inputs and draw predicted class names in each frame.

#### Requirements
* Python 3
* Pillow
* ffmpeg, ffprobe

#### Usage
To generate videos based on ```../output.json```, execute the following.
```
python generate_result_video.py ../output.json ../videos ./videos_pred ../class_names_list 5
```

The 2nd parameter (```../videos```) is the root directory of videos.
The 3rd parameter (```./videos_pred```) is the directory path of output videos.
The 5th parameter is a size of temporal unit.  
The CNN predicts class scores for a 16 frame clip.  
The code averages the scores over each unit.  
The size 5 means that it averages the scores over 5 clips (i.e. 16x5 frames).  
If you use the size as 0, the scores are averaged over all clips of a video. 

To generate a text-only result:
```
./text_results.sh
```

Full list of parameters:


     '--input'				default='input', type=str, help='Input file path'
     '--video_root'				default='', type=str, help='Root path of input videos'
     '--model'				default='', type=str, help='Model file path'
     '--output'				default='output.json', type=str, help='Output file path'
     '--mode'				default='score', type=str, help='Mode (score | feature). score outputs class scores. feature outputs features (after global average pooling).'
     '--batch_size'				default=32, type=int, help='Batch Size'
     '--n_threads'				default=4, type=int, help='Number of threads for multi-thread loading'
     '--model_name'				default='resnet', type=str, help='Currently only support resnet'
     '--model_depth'			default=34, type=int, help='Depth of resnet (10 | 18 | 34 | 50 | 101)'
     '--resnet_shortcut'		        default='A', type=str, help='Shortcut type of resnet (A | B)'
     '--wide_resnet_k'			default=2, type=int, help='Wide resnet k'
     '--resnext_cardinality'	        default=32, type=int, help='ResNeXt cardinality'
     '--no_cuda'				help='If true, cuda is not used.'
     '--verbose'				


## 10. Results

* Some examples of videos captioned using this code:
[TennisSwing](https://www.youtube.com/watch?v=OGxrwzY-aDw)
[Typing](https://www.youtube.com/watch?v=d19IbrQS6eE)
[TrampolineJumping](https://www.youtube.com/watch?v=TPnp-UjrCII)



# Video Classification Using 3D ResNet (original readme)
This is a pytorch code for video (action) classification using 3D ResNet trained by [this code](https://github.com/kenshohara/3D-ResNets-PyTorch).  
The 3D ResNet is trained on the Kinetics dataset, which includes 400 action classes.  
This code uses videos as inputs and outputs class names and predicted class scores for each 16 frames in the score mode.  
In the feature mode, this code outputs features of 512 dims (after global average pooling) for each 16 frames.  

**Torch (Lua) version of this code is available [here](https://github.com/kenshohara/video-classification-3d-cnn).**

## Requirements
* [PyTorch](http://pytorch.org/)
```
conda install pytorch torchvision cuda80 -c soumith
```
* FFmpeg, FFprobe
```
wget http://johnvansickle.com/ffmpeg/releases/ffmpeg-release-64bit-static.tar.xz
tar xvf ffmpeg-release-64bit-static.tar.xz
cd ./ffmpeg-3.3.3-64bit-static/; sudo cp ffmpeg ffprobe /usr/local/bin;
```
* Python 3

## Preparation
* Download this code.
* Download the [pretrained model](https://drive.google.com/drive/folders/14KRBqT8ySfPtFSuLsFS2U4I-ihTDs0Y9?usp=sharing).  
  * ResNeXt-101 achieved the best performance in our experiments. (See [paper](https://arxiv.org/abs/1711.09577) in details.)

## Usage
Assume input video files are located in ```./videos```.

To calculate class scores for each 16 frames, use ```--mode score```.
```
python main.py --input ./input --video_root ./videos --output ./output.json --model ./resnet-34-kinetics.pth --mode score
```
To visualize the classification results, use ```generate_result_video/generate_result_video.py```.

To calculate video features for each 16 frames, use ```--mode feature```.
```
python main.py --input ./input --video_root ./videos --output ./output.json --model ./resnet-34-kinetics.pth --mode feature
```


## Citation
If you use this code, please cite the following:
```
@article{hara3dcnns,
  author={Kensho Hara and Hirokatsu Kataoka and Yutaka Satoh},
  title={Can Spatiotemporal 3D CNNs Retrace the History of 2D CNNs and ImageNet?},
  journal={arXiv preprint},
  volume={arXiv:1711.09577},
  year={2017},
}
```
