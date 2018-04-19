# Recurrent 3D Pose Sequence Machines

[Paper Link](http://openaccess.thecvf.com/content_cvpr_2017/papers/Lin_Recurrent_3D_Pose_CVPR_2017_paper.pdf)
[Human 3.6M Dataset Link](https://pan.baidu.com/s/1bpvSLBp#list/path=%2F)

## Setup
- Install [Anaconda](https://conda.io/docs/user-guide/install/download.html)
- Create a conda environment: `conda env create -f requirements.txt -n <ENVIRONMENT NAME>`
- Activate the environment: `source activate <ENVIRONMENT NAME>`
- Add the uvic repo: `git remote add git@gitlab.csc.uvic.ca:courses/201801/csc486b/final-project/group-c/term-project.git`
- Set the uvic repo as an origin branch: `git remote set-url --add --push origin git@gitlab.csc.uvic.ca:courses/201801/csc486b/final-project/group-c/term-project.git`
- Set the github repo as an origin branch (to mirror the codebase): `git remote set-url --add --push origin git@github.com:andrewandrewxu/CSC486_Project.git`

Authors: Christopher Brett, Tyson Battistella, Shae Brown, Andrew Xu
Academic Institution: Univeristy of Victoria 
Year: 2018
Course: Computer Science 486 Spring 2018

## Training the model

Before you can run the model, the [Human 3.6M Dataset](http://vision.imar.ro/human3.6m/description.php) must be downloaded and processed following these [instructions](https://github.com/MudeLin/RPSM/tree/master/util/preprocess)

To bypass these steps, we provide the following pre-processed data and meta data:
[Images and image list](https://drive.google.com/drive/folders/1LkQCl6rSXiOE7JOo9DVhrhGvDOBdCQrZ)
[HDF5 files](https://drive.google.com/drive/folders/1oD_0DjI04ECwJSEUTQP0lBuGf-rxILgf)

The model can be trainined by running main.py with the following arguments:

```
--data_dir {hdf5 folder}
--nt_iters {number of iterations}
--train_h5_path {training hdf5 file}
--valid_h5_path {validation hdf5 file}
--root_image_folder {folder for the images}
--train_image_list {image list for training}
--valid_image_list {image list for validation}
--max_frames {number of frames}
```

Run main.py -h for more information about the arguments.
