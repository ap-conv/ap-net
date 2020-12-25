# ap-net
The pytorch implementation for "Image Recognition with Augmentation Pathways".

## Requirements
1. Python 3.7
2. PyTorch 1.3.0
3. [pytorch-randaugment](https://github.com/ildoonet/pytorch-randaugment)

## Dataset Prepare
1. Download ImageNet dataset to folder `datasets/`
2. Data Organization:
   - All images are supposed to be in `datasets/imagenet/all`
   - Train/Valid annotations should be in `datasets/imagenet/` and named as `(train|valid).txt`
   - Annotation file format:
   ```
   name_of_image.jpg label_num_from_zero\n
   ```

## Training
For ImageNet dataset, you can train AP-IResNet-50 from scratch with following script:
```bash
python train.py -M 9 -N 2 --desc AP-IResNet-50
```

## Results
| Model         |  #Params    |  MACs    |  Augmentation    |  Acc          |
|---------------|:-----------:|:--------:|:----------------:|:-------------:|
| iResNet-50    |  25.6M      |  4.15G   |  Baseline        |  77.59/93.55  |
| iResNet-50    |  25.6M      |  4.15G   |  RandAugment     |  77.20/93.52  |
| AP-iResNet-50 |  21.8M      |  3.95G   |  RandAugment     |  78.20/93.95  |