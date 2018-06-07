# FBSuperResolution

The implementation of multi-frame super resolution via sub-pixel. All reference are following.

## Usage

1. clone or download this repo to local
```
git clone git@github.com:zhangxiaoya/FB.git
```
2. Make foler for build
```
cd FB
mkdir build
cd build
cmake .. # or use cmake-gui for custom build example or not, default is build with example
make
```
3. Test SR use example
```
./example/example_runtime
```
> the low resolution images are store at data folder by default, and the high resolution result is store at result folder by default.

4. Result

![](https://github.com/zhangxiaoya/FB/blob/master/result/eia_4*4_result_00.png)

## References

1. [Fast and Robust Multiframe Super Resolution](https://www.semanticscholar.org/paper/Fast-and-robust-multiframe-super-resolution-Farsiu-Robinson/61997bb7d5a041353582599caf52fd5014cf60cb)
2. [Pyramidal Implementation of the Lucas Kanade Feature Tracker, description of the algorithm](http://robots.stanford.edu/cs223b04/algo_tracking.pdf)
3. [MDSP Super-Resolution And Demosaicing Datasets](https://users.soe.ucsc.edu/~milanfar/software/sr-datasets.html)
