## how to use this codebase? 

cd /Users/johnny/Desktop/ierg4190_project


mkdir -p .tmp/out
TMPDIR=$PWD/.tmp /Users/johnny/miniconda3/envs/ml/bin/python run.py \
  --data-dir src/data_tv --output-dir .tmp/out

You should get something like this : 

{"noise_std": 0.1, "filter": "none", "feature": "pixels", "classifier": "svm", "seed": 42, "psnr": 100.0, "mse": 0.0, "accuracy": 0.0383, "macro_f1": 0.03941504822552364}


## How to do noise sweep 


cd /Users/johnny/Desktop/ierg4190_project


mkdir -p .tmp/out
TMPDIR=$PWD/.tmp /Users/johnny/miniconda3/envs/ml/bin/python run.py --run-all\
  --data-dir src/data_tv --output-dir .tmp/out

You should get a list of json data that shows how noise  affect classification result
{"noise_std": 0.0, "filter": "none", "feature": "pixels", "classifier": "svm", "seed": 42, "psnr": 100.0, "mse": 0.0, "accuracy": 0.0383, "macro_f1": 0.03941504822552364}
{"noise_std": 0.05, "filter": "none", "feature": "pixels", "classifier": "svm", "seed": 42, "psnr": 74.2, "mse": 0.0, "accuracy": 0.0391, "macro_f1": 0.0401050527442493}
{"noise_std": 0.1, "filter": "none", "feature": "pixels", "classifier": "svm", "seed": 42, "psnr": 68.18, "mse": 0.0, "accuracy": 0.0405, "macro_f1": 0.04146148751172161}
{"noise_std": 0.2, "filter": "none", "feature": "pixels", "classifier": "svm", "seed": 42, "psnr": 62.16, "mse": 1e-06, "accuracy": 0.0408, "macro_f1": 0.04185996823033011}
{"noise_std": 0.3, "filter": "none", "feature": "pixels", "classifier": "svm", "seed": 42, "psnr": 58.64, "mse": 1e-06, "accuracy": 0.0382, "macro_f1": 0.03828610366861531}