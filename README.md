# CoNet
Co-Net: A Collaborative Region-Contour-Driven Network for Fine-to-Finer Medical Image Segmentation (WACV 2022)

## 1. Training/Testing:
The training and testing experiments are conducted using PyTorch with NVIDIA TITAN Xp GPU

Installing necessary packages: PyTorch 1.1
* downloading training and testing GEDD datasets and move it into ./data/Train and ./data/Test which can be found in this [Download link (Google Drive)](https://drive.google.com/drive/folders/17E1N9TOt4G96ynwW9i6TpQpkiMxKxTh4)
* downloading training and testing polyp datasets and move it into ./data/TrainDataset and ./data/TestDataset which can be found in this [Download link (Google Drive)](https://drive.google.com/drive/folders/17E1N9TOt4G96ynwW9i6TpQpkiMxKxTh4)
* downloading Res2Net weights in [Download link (Google Drive)](https://drive.google.com/file/d/1747Tn5ws00IPlgt1lhCTIpkUIqaIs3VU/view)
* Assigning your costumed path, like --train_save and --train_path in Train.py
* Training the CoNet as python Train.py
* Running Test.py to generate the final prediction map: replace your trained model directory (--pth_path). One pretrained weights of Co-Net can be found in [Download link (Google Drive)](https://drive.google.com/drive/folders/160pacX7qxLW3bbuzR7DvjlrQlP93kyN0)

## 2. Evaluating your trained model:
* The evaluation code is written in MATLAB, just run ./eval/main.py to generate the evaluation results in ./results/
