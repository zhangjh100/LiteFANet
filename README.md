# LiteFANet
Please prepare an environment with python=3.8

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


