[![Board Status](https://zhangxiaoya1989.visualstudio.com/6c7ceafb-920b-48d9-9e3a-7cd936385321/a7b61938-c61f-4b06-91e3-71fbb8ac64ec/_apis/work/boardbadge/62cc233f-3d09-4ab1-a549-abf8ad4fe3f6)](https://zhangxiaoya1989.visualstudio.com/6c7ceafb-920b-48d9-9e3a-7cd936385321/_boards/board/t/a7b61938-c61f-4b06-91e3-71fbb8ac64ec/Microsoft.RequirementCategory)
# FBSuperResolution

The implementation of multi-frame super resolution via sub-pixel. All reference are following.

## Usage

1. clone or download this repo to local
```
git clone git@github.com:zhangxiaoya/FB.git
```
2. Make foler for build

**Use OpenCV2**

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

![原始图像](https://github.com/zhangxiaoya/FB/blob/master/data/eia/000000.png)
![超分辨率图像](https://github.com/zhangxiaoya/FB/blob/master/result/eia_4*4_result_00.png)

## References

1. [Fast and Robust Multiframe Super Resolution](https://www.semanticscholar.org/paper/Fast-and-robust-multiframe-super-resolution-Farsiu-Robinson/61997bb7d5a041353582599caf52fd5014cf60cb)
2. [Pyramidal Implementation of the Lucas Kanade Feature Tracker, description of the algorithm](http://robots.stanford.edu/cs223b04/algo_tracking.pdf)
3. [MDSP Super-Resolution And Demosaicing Datasets](https://users.soe.ucsc.edu/~milanfar/software/sr-datasets.html)
