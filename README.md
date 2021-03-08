



# Self-supervised point set local descriptors for point cloud registration. 

![pipeline](https://github.com/STAR-Center/selfSupervisedDescriptor/blob/master/docs/pipeline.jpg)

Yuan, Y., Borrmann, D., Hou, J., Ma, Y., Nüchter, A., & Schwertfeger, S. (2021). Self-supervised point set local descriptors for point cloud registration. Sensors, 21(2), 486.


### Environment

Our code is developed on top of the implementation of [3DFeatNet](https://github.com/yewzijian/3DFeatNet).

Please follow [3DFeatNet](https://github.com/yewzijian/3DFeatNet) to preprocess the data.


####  Repo structure
```
./use_descriptor
-> train.sh # bash script to train model
-> exp/ # bash scripts to directly run experiments
-> models/ # network
-> tf_ops/ # PN clustering operation 
-> cpp/ # preprocess including ISS-keypoints and FPFH-keypoints (not used)
-> data/ # data augmentation and loading
-> ckpt/ # model checkpoints
-> inference.py
-> train1jit.py
-> config.py
```




