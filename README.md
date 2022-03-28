This repo is for our paper "Data Imputation for Multivariate Time Series Sensor Data with Large Gaps of Missing Data." The LGDI algorithm is demoed in a jupyter-notebook.

## Quick Start
First, we need to install a Python 3 virtual environment with:
```
sudo apt-get install python3-venv
```

Create a virtual environment:
```
python3 -m venv python_venv
```

You need to activate the virtual environment when you want to use it:
```
source python_venv/bin/activate
```

To fufil all the requirements for the python server, you need to run:
```
pip3 install -r requirements.txt
```
Because we are now inside a virtual environment. We do not need sudo.

Then you can start the Jupyter-Notebook server with:
```
jupyter-notebook
```

All the sample data is stored in the data folder.