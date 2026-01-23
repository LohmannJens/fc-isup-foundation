# Federated ISUP grade prediction with foundation models FeatureCloud App

## Description
A FeatureCloud app for communication efficient prediction of ISUP grades [1] using foundation models from tissue microarray (TMA) images.
It was developed based on the first step of eCaReNet [2] for prostate cancer relapse
over time prediction.


## Input
In each client the following folders and files must be given:
- `train.csv`: local training data, needs to include embeddings of image and ISUP grade
- `valid.csv`: local validation data, needs to include embeddings of image and ISUP grade
- `test.csv`: local test data, needs to to include embeddings of image and ISUP grade

Over all clients these files are the same:
- `config.yaml`: config file with information about the trainig

The input images should be embedded by a foundation model.
We used UNI [3] and CONCH [4] for that but others are also possible (but not tested).


## Output
- `results.csv`: predicted class per image and metadata for the test dataset
- `confusion_matrix.png`: confusion matrix of the test dataset predictions (absolute values)


## Workflows
Combinations with other FeatureCloud apps was not tested yet.


## Config
Use the config file to customize your training and evaluation.
Needs to be uploaded together with the training data as `config.yaml`
```
input_channels: # int, number of input_channels for the classification layer (equals embedding space).
lr: # float, learning rate to use for the optimizer (SGD is implemented right now).
epochs: # int, number of epochs to train the model. We assume number of epochs == number of communcations
batch_size: # int, batch size to be used for loading the datasets.
use_smpc: # bool, indicate if secure multi party compuation (SMPC) should be used.

```

## Test data
For testing the functionality of the app we provide dummy data in `data/testdata`.
The dummy test data includes two clients with images embedded by the UNI model [3].
Please refere to the test data to check for the structure of the input data


## References
[1] Egevad, L., Delahunt, B., Srigley, J. R., & Samaratunga, H. (2016). International Society of Urological Pathology (ISUP) grading of prostate cancer – An ISUP consensus on contemporary grading. APMIS, 124(6), 433–435. https://doi.org/10.1111/apm.12533

[2] Dietrich, E., Fuhlert, P., Ernst, A., Sauter, G., Lennartz, M., Stiehl, H. S., Zimmermann, M., & Bonn, S. (2021). Towards Explainable End-to-End Prostate Cancer Relapse Prediction from H&E Images Combining Self-Attention Multiple Instance Learning with a Recurrent Neural Network. In S. Roy, S. Pfohl, E. Rocheteau, G. A. Tadesse, L. Oala, F. Falck, Y. Zhou, L. Shen, G. Zamzmi, P. Mugambi, A. Zirikly, M. B. A. McDermott, & E. Alsentzer (Eds.), Proceedings of Machine Learning for Health (Vol. 158, pp. 38–53). PMLR. https://proceedings.mlr.press/v158/dietrich21a.html

[3] Chen, R. J., Ding, T., Lu, M. Y., Williamson, D. F. K., Jaume, G., Song, A. H., Chen, B., Zhang, A., Shao, D., Shaban, M., Williams, M., Oldenburg, L., Weishaupt, L. L., Wang, J. J., Vaidya, A., Le, L. P., Gerber, G., Sahai, S., Williams, W., & Mahmood, F. (2024). Towards a general-purpose foundation model for computational pathology. Nature Medicine, 30(3), 850–862. https://doi.org/10.1038/s41591-024-02857-3


[4] Lu, M. Y., Chen, B., Williamson, D. F. K., Chen, R. J., Liang, I., Ding, T., Jaume, G., Odintsov, I., Le, L. P., Gerber, G., Parwani, A. V., Zhang, A., & Mahmood, F. (2024). A visual-language foundation model for computational pathology. Nature Medicine, 30(3), 863–874. https://doi.org/10.1038/s41591-024-02856-4
