# LiteFANet
Please prepare an environment with python=3.8

```
pip install -r requirements.txt
```

# 3D CBCT Tooth dataset:

```
./datasets/3D-CBCT-Tooth/
    sub_volumes/160-160-96_2048.npz
    train/
        images/
            1000889125_20171009.nii.gz
            ......
            X2360674.nii.gz
        labels/
            1000889125_20171009.nii.gz
            ......
            X2360674.nii.gz
    valid/
        images/
            1000813648_20180116.nii.gz
            ......
            X2358714.nii.gz
        labels/
            1000813648_20180116.nii.gz
            ......
            X2358714.nii.gz
```

# MMOTU dataset:

```
./datasets/MMOTU/
	train/
		images/
			1.JPG
			......
			1465.JPG
		labels/
			1.PNG
			......
			1465.PNG
	valid/
		images/
			3.JPG
			......
			1469.JPG
		labels/
			3.PNG
			......
			1469.PNG
```


# ISIC-2018 dataset:

```
./datasets/ISIC-2018/
	train/
		images/
			ISIC_0000000.jpg
			......
			ISIC_0016072.jpg
		annotations/
			ISIC_0000000_segmentation.png
			......
			ISIC_0016072_segmentation.png
	test/
		images/
			ISIC_0000003.jpg
			......
			ISIC_0016060.jpg
		annotations/
			ISIC_0000003_segmentation.png
			......
			ISIC_0016060_segmentation.png
```



# Training demo:
```
--dataset: dataset name, optional 3D-CBCT-Tooth, MMOTU, ISIC-2018
--model: model name, see below implemented architectures for details
--dimension: dimension of dataset images and models, for LieFANet only
--scaling_version: Ultra, Pro, Basic
--epoch: training epoch
```

```
python ./train.py --dataset 3D-CBCT-Tooth --model LieFANet --dimension 3d --scaling_version Basic --epoch 20
python ./train.py --dataset MMOTU --model LieFANet --dimension 2d --scaling_version Ultra --epoch 2000
python ./train.py --dataset ISIC-2018 --model LieFANet --dimension 2d --scaling_version Ultra --epoch 150
```

# Testing demo:
```
python ./test.py --dataset 3D-CBCT-Tooth --model LieFANet --pretrain_weight ./pretrain/LieFANet3D-TINY_Tooth.pth --dimension 3d --scaling_version Basic
python ./test.py --dataset MMOTU --model LieFANet --pretrain_weight ./pretrain/LieFANet2D-BASIC_MMOTU.pth --dimension 2d --scaling_version Ultra
python ./test.py --dataset ISIC-2018 --model LieFANet --pretrain_weight ./pretrain/LieFANet2D-BASIC_ISIC2018.pth --dimension 2d --scaling_version Ultra
```


# Inferenceing demo:
```
python ./inference.py --dataset 3D-CBCT-Tooth --model LieFANet --pretrain_weight ./pretrain/LieFANet3D-TINY_Tooth.pth --dimension 3d --scaling_version Basic --image_path ./images/1001250407_20190923.nii.gz
python ./inference.py --dataset MMOTU --model LieFANet --pretrain_weight ./pretrain/LieFANet2D-BASIC_MMOTU.pth --dimension 2d --scaling_version Ultra --image_path ./images/453.JPG
python ./inference.py --dataset ISIC-2018 --model LieFANet --pretrain_weight ./pretrain/LieFANet2D-BASIC_ISIC2018.pth --dimension 2d --scaling_version Ultra --image_path ./images/ISIC_0000550.jpg
```
