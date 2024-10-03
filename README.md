# yolo-sam

I use a conda env with python 3.10

For information, Ultralytics version of this repo is `8.2.97`, Segment-Anything-2 commit is `52198ead0eb13ae8270bea6ca768ef175f5bf167` (version `2.1.0`)

## Installation

Install dependencies:
`pip install -r requirements.txt`

### Start MLflow server

On a new command prompt:

``mlflow server --host 127.0.0.1 --port 1234``

### Start a training

#### Training Settings

Update settings training on `ultralytics/train.yaml` document.

#### Training

To start a training:

```
cd ultralytics
python train.py
```

> Note that the folder containing the runs must be in a folder whose name begins with "mlflow-"