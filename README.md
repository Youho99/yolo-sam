# yolo-sam

I use a conda env with python 3.10

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