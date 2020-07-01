# MS-TCN2

## Environment
Python3

## Training:
Download the data folder, which contains the features and the ground truth labels. (~30GB) (If you cannot download the data from the previous link, try to download it from here)
Extract it so that you have the data folder in the same directory as main.py.
To train the model run sh train.sh ${dataset} ${split} where ${dataset} is breakfast, 50salads or gtea, and ${split} is the split number (1-5) for 50salads and (1-4) for the other datasets.

## Evaluation
Run sh test_epoch.sh ${dataset} ${split} ${test_epoch}.

