# XG Boost Benchmarking

> e**X**treme **G**radient **Boost**ing, is an open-source machine learning library built for an efficient implementation of distributed, gradient boosted tree-based algorithms.

The XGBoost algorithm was primarily designed for efficiency of compute time and memory resources. XGBoost delivers parallel-tree boosting and focuses on machine learning techniques such as regression, classification, and ranking problems.

## Installing and using XGBoost optimizations

### Installing Anaconda

Prerequisites

```shell
sudo apt-get install libgl1-mesa-glx libegl1-mesa libxrandr2 libxrandr2 libxss1 libxcursor1 libxcomposite1 libasound2 libxi6 libxtst6
```

Download the Anaconda installer for Linux.

```shell
wget https://repo.anaconda.com/archive/Anaconda3-2022.10-Linux-x86_64.sh
```

Install Anaconda

```shell
bash ~/path/to/Anaconda3-2022.10-Linux-x86_64.sh
```

- Press `Enter` to review the license agreement. Then press and hold Enter to scroll.

- Enter `“yes”` to agree to the license agreement.

- The installer prompts you to choose whether to initialize Anaconda Distribution by running conda init. Anaconda recommends entering `“yes”`

- The installer finishes and displays, “Thank you for installing Anaconda<2/3>!”

- Close and re-open your terminal window for the installation to take effect, or enter the command `source ~/.bashrc` to refresh the terminal.

### Download Dataset to benchmark

Download the Higgs Dataset

```shell
wget https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz
```

### Benchmarking

Run the file [`XGBoost.ipynb`](./XGBoost.ipynb) in different environments

- Unoptimised older XGBoost v0.8.1

- XGBoost v1.4.2 without Intel Optimised libraries

- Intel `intel-aikit` which contains Intel Optimised version of XGBoost v1.4.2

- Latest XGBoost v1.7.1

### Creating and Enabling Environments

#### 1. For Intel `intel-aikit` which contains Intel Optimised version of XGBoost v1.4.2

Create the Conda Virtual Environment

```shell
conda create -n optimised-xgboost -c intel intel-aikit-modin
```

Activate environment

```shell
conda activate optimised-xgboost
```

Install required libraries through pip

```shell
pip install -r ./Requirements/Intel-optimised-xgboost-pip.txt
```

Now that environment is setup, Run the [Ipynb file](./XGBoost.ipynb)

Deactivate environment

```shell
conda deactivate optimised-xgboost
```

#### 2. For Unoptimised older XGBoost v0.8.1

Create the Conda Virtual Environment

```shell
conda create --name xgboostv0.8.1 --clone optimised-xgboost
```

Activate environment

```shell
conda activate xgboostv0.8.1
```

Install required libraries through pip

```shell
# Removing existing version of xgboost
conda remove xgboost

pip install -r ./Requirements/xgboost-v0.8.1-pip.txt
```

Now that environment is setup, Run the [Ipynb file](./XGBoost.ipynb)

Deactivate environment

```shell
conda deactivate xgboostv0.8.1
```

#### 3. XGBoost v1.4.2 without Intel Optimised libraries

Create the Conda Virtual Environment

```shell
conda create --name xgboostv1.4.2 pip
```

Activate environment

```shell
conda activate xgboostv1.4.2
```

Install required libraries through pip

```shell
pip install -r ./Requirements/xgboost-v1.4.2-pip.txt
```

Now that environment is setup, Run the [Ipynb file](./XGBoost.ipynb)

Deactivate environment

```shell
conda deactivate xgboostv1.4.2
```

#### 4. Latest XGBoost v1.7.1

Create the Conda Virtual Environment

```shell
conda create --name xgboost-latest pip
```

Activate environment

```shell
conda activate xgboost-latest
```

Install required libraries through pip

```shell
pip install -r ./Requirements/xgboost-latest-pip.txt
```

Now that environment is setup, Run the [Ipynb file](./XGBoost.ipynb)

Deactivate environment

```shell
conda deactivate xgboost-latest
```

# XG Boost vs daal4py Benchmarking

For Python 3.6 and higher versions, the inference speedup attained by XGBoost optimized by Intel can further be elevated using the daal4py API of [Intel® oneDAL library](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onedal.html).

## Benchmarking Intel Optimised XGBoost vs daal4py

Activate the virtual environment in which Intel optimised XGBoost exists.

```shell
conda activate optimised-xgboost
```

Install the daal4py API:

```shell
pip install daal4py
```

If there is any error installing daal4py, run

```shell
pip install daal4py --ignore-installed
```

Now run the [Ipynb File](./daal4py.ipynb) for benchmarking XGBoost vs daal4py

### References

- [Intel AIKit Modin](https://www.intel.com/content/www/us/en/develop/documentation/installation-guide-for-intel-oneapi-toolkits-linux/top/installation/install-using-package-managers/conda/install-intel-ai-analytics-toolkit-via-conda.html)

- [XG Boost Benchmarking](https://github.com/oneapi-src/oneAPI-samples/blob/master/AI-and-Analytics/Features-and-Functionality/IntelPython_XGBoost_Performance/IntelPython_XGBoost_Performance.ipynb)

- [XGBoost vs daal4py](https://github.com/oneapi-src/oneAPI-samples/blob/master/AI-and-Analytics/Features-and-Functionality/IntelPython_XGBoost_daal4pyPrediction/IntelPython_XGBoost_daal4pyPrediction.ipynb)
