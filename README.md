# MS-TCN++: Multi-Stage Temporal Convolutional Network for Action Segmentation (TPAMI 2020)

This repository provides a PyTorch implementation of the paper [MS-TCN++: Multi-Stage Temporal Convolutional Network for Action Segmentation](https://arxiv.org/pdf/2006.09220.pdf).

## Environment
Python3, pytorch

## Training:
* Download the [data](https://mega.nz/#!O6wXlSTS!wcEoDT4Ctq5HRq_hV-aWeVF1_JB3cacQBQqOLjCIbc8) folder, which contains the features and the ground truth labels. (~30GB) (If you cannot download the data from the previous link, try to download it from [here](https://zenodo.org/record/3625992#.Xiv9jGhKhPY))
* Extract it so that you have the `data` folder in the same directory as `main.py`.
* To train the model run sh train.sh ${dataset} ${split} where ${dataset} is breakfast, 50salads or gtea, and ${split} is the split number (1-5) for 50salads and (1-4) for the other datasets.

## Evaluation
Run sh test_epoch.sh ${dataset} ${split} ${test_epoch}.


## Cite:
```BibTeX
@article{li2020ms,
   author={Shi-Jie Li and Yazan AbuFarha and Yun Liu and Ming-Ming Cheng and Juergen Gall},
    journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
    title={MS-TCN++: Multi-Stage Temporal Convolutional Network for Action Segmentation}, 
    year={2020},
    volume={},
    number={},
    pages={1-1},
    doi={10.1109/TPAMI.2020.3021756},
}

@inproceedings{farha2019ms,
  title={Ms-tcn: Multi-stage temporal convolutional network for action segmentation},
  author={Farha, Yazan Abu and Gall, Jurgen},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={3575--3584},
  year={2019}
}

```
