# LSTM-AE SageMaker Algorithm
The [Time Series Anomaly Detection (LSTM-AE) Algorithm from AWS Marketplace](placeholder) 
performs time series anomaly detection with a Long Short-Term Memory Network Autoencoder (LSTM-AE).
It implements both training and inference from CSV data and supports both CPU and GPU instances.
The training and inference Docker images were built by extending the PyTorch 2.0 Python 3.10 SageMaker containers.

## Model Description
### Model Architecture
The LSTM-AE model reconstructs the time series with a multivariate LSTM autoencoder.
The encoder and decoder are stacked LSTM models with the same number of layers.
Each LSTM layer has the same number of hidden units, and therefore the autoencoder does not have the usual pyramidal structure.
The encoder takes as input the time series and learns a one-dimensional latent representation.
The decoder takes as input the one-dimensional latent representation and learns to reconstruct the time series. 
The autoencoder parameters are learned on a training set containing only normal data (i.e. without anomalies) by minimizing the mean squared error (MSE) between the actual and reconstructed values of the time series.

<img src=https://fg-research-assets.s3.eu-west-1.amazonaws.com/lstm-ae-diagram.png style="width:60%;margin-top:60px;margin-bottom:50px"/>

*LSTM-AE architecture (source: [doi: 10.1109/ATIT49449.2019.9030505](https://ieeexplore.ieee.org/document/9030505))*

**Model Resources:** [[Paper]](https://ieeexplore.ieee.org/document/9030505)

## SageMaker Algorithm Description
The algorithm implements the model as described above with no changes.

### Training
The training algorithm has two input data channels: `training` and `validation`. 
The `training` channel is mandatory, while the `validation` channel is optional.

The training and validation datasets should be provided as CSV files and should only contain normal data (i.e. without anomalies).
Each column of the CSV file represents a time series, while each row represents a time step.
All the time series should have the same length and should not contain missing values.
The CSV file should not contain any index column or column headers.

See the sample input files `train.csv` and `valid.csv` in the `data/training/` folder.
See `notebook.ipynb` for an example of how to launch a training job.

#### Distributed Training
The algorithm supports multi-GPU training on a single instance, which is implemented through [torch.nn.DataParallel](https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html).
The algorithm does not support multi-node (or distributed) training across multiple instances. 

#### Hyperparameters
The training algorithm takes as input the following hyperparameters:
- `sequence-length`: `int`. The length of the sequences.
- `sequence-stride`: `int`. The period between consecutive sequences.
- `num-layers`: `int`. The number of LSTM layers.
- `hidden-size`: `int`. The number of hidden units of each LSTM layer.
- `dropout`: `float`. The dropout rate applied after each LSTM layer.
- `lr`: `float`. The learning rate used for training.
- `batch-size`: `int`. The batch size used for training.
- `epochs`: `int`. The number of training epochs.

#### Metrics
The training algorithm logs the following metrics:
- `train_mse`: `float`. Training mean squared error.
- `train_mae`: `float`. Training mean absolute error.

If the validation channel is provided, the training algorithm also logs the following additional metrics:
- `valid_mse`: `float`. Validation mean squared error.
- `valid_mae`: `float`. Validation mean absolute error.

See `notebook.ipynb` for an example of how to launch a hyperparameter tuning job.

### Inference
The inference algorithm takes as input a CSV file containing the time series.
Each column of the CSV file represents a time series, while each row represents a time step.
The CSV file should not contain any index column or column headers.
All the time series should have the same length and should not contain missing values.
See the sample input file `test.csv` in the `data/inference/input` folder.

The inference algorithm outputs the reconstructed values of the time series.
See the sample output file `batch_predictions.csv` in the `data/inference/output/batch` folder.
See `notebook.ipynb` for an example of how to launch a batch transform job.

#### Endpoints
The algorithm supports only real-time inference endpoints. The inference image is too large to be uploaded to a serverless inference endpoint.

See `notebook.ipynb` for an example of how to deploy the model to an endpoint, invoke the endpoint and process the response.
See the sample output file `real_time_predictions.csv` in the `data/inference/output/real-time` folder.

## References
- O. I. Provotar, Y. M. Linder and M. M. Veres, "Unsupervised Anomaly Detection in Time Series Using LSTM-Based Autoencoders," *2019 IEEE International Conference on Advanced Trends in Information Theory (ATIT)*, Kyiv, Ukraine, 2019, pp. 513-517, [doi: 10.1109/ATIT49449.2019.9030505](https://ieeexplore.ieee.org/document/9030505).