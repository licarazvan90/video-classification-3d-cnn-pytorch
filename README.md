

# Action recognition using 3D-Res-Net: https://github.com/kenshohara/video-classification-3d-cnn-pytorch


##1. Install prerequisites (Debian 9): 
sudo apt install bzip2, gcc, make, cmake, linux-source, linux-headers-`uname -r`, git


##2. Install Anaconda

wget https://repo.anaconda.com/archive/Anaconda3-5.1.0-Linux-x86_64.sh
chmod +x Anaconda3-5.1.0-Linux-x86_64.sh
./Anaconda3-5.1.0-Linux-x86_64.sh
source .bashrc

##3. Get nvidia driver:
wget http://us.download.nvidia.com/XFree86/Linux-x86_64/384.66/NVIDIA-Linux-x86_64-384.66.run
chmod +x NVIDIA-Linux-x86_64-340.65.run
sudo ./NVIDIA-Linux-x86_64-340.65.run

##4. Install PyTorch and CUDA
conda install pytorch torchvision cuda80 -c soumith

##5. FFmpeg, FFprobe
wget http://johnvansickle.com/ffmpeg/releases/ffmpeg-release-64bit-static.tar.xz
tar xvf ffmpeg-release-64bit-static.tar.xz
cd ./ffmpeg-3.3.3-64bit-static/; sudo cp ffmpeg ffprobe /usr/local/bin;

##6. GET the 3d-res-net repository
git clone https://github.com/kenshohara/video-classification-3d-cnn-pytorch.git


##7. Download model from google drive:

Add this to the .bashrc file:

function gdrive_download () {
                  CONFIRM=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://docs.google.com/uc?export=download&id=$1" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')
                              wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$CONFIRM&id=$1" -O $2
                                            rm -rf /tmp/cookies.txt
                            }

Then:

source .bashrc
cd video-classification-3d-cnn-pytorch/models
gdrive_download 1NOSyEnwPuEdtWHnsVC8JrDfL0sY3iKZV resnext-101-kinetics.pth


##8. Download some video files from here (THUMOS14):
http://crcv.ucf.edu/THUMOS14/test_set/TH14_test_set_mp4/

mkdir video-classification-3d-cnn-pytorch/videos
wget http://crcv.ucf.edu/THUMOS14/test_set/TH14_test_set_mp4/video_test_0000002.mp4


##9. Run the CODE:



###9.1 Process files
Assume input video files are located in ./videos. And the 'input' file contains the names of the video files there.
To calculate class scores for each 16 frames, use --mode score.

python main.py --input ./input --video_root ./videos --output ./output.json --model ./resnet-34-kinetics.pth --mode score


###9.2 Result Video Generation
This is a code for generating videos of classification results.  
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

To output the results:
python generate_result_video/generate_result_video.py ./output.json ./videos/ ./results ./class_names_list 5
























# (original readme) Video Classification Using 3D ResNet
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
