# Environment Setup

Week 1 Introduction to Machine Learning and Environment Setup

## Linux Environment

* Using Windows Subsystem for Linux (WSL) on local Windows 11 computer
* Resources on setting it up can be found [here](https://learn.microsoft.com/en-us/windows/wsl/setup/environment)

## Python

### Install

* Using Anaconda which can be downloaded [here](https://www.anaconda.com/download)
* Copy linux download link
* Install 
    * `wget https://repo.anaconda.com/archive/Anaconda3-2023.07-2-Linux-x86_64.sh`
* You will see: Saving to: ‘Anaconda3-2023.07-2-Linux-x86_64.sh’
* Bash 
    * `bash Anaconda3-2023.07-2-Linux-x86_64.sh`
* Accept the license terms
* initialize

### Environment & Libraries

* create environment named ml-zoomcamp with python version 3.9 
    * `conda create -n ml-zoomcamp python=3.9`
* activate environment
    * `conda activate ml-zoomcamp`
* install libraries
    * `conda install numpy pandas scikit-learn seaborn jupyter`
* to leave an environment
    * `conda deactivate`

## Docker

When using WSL you just have to download Docker Desktop on Windows and it will be used in the WSL without needing to install Docker.io.

* Docker install link and instruction can be found [here](https://docs.docker.com/desktop/install/windows-install/)
* Run installer .exe 
* Ensure WSL 2 backend is being used in General Settings on Docker Desktop GUI

## Random

* Using Google Colab you can run Jupyter notebooks on Github
* replace the 'https://github.com/' portion with 'https://colab.research.google.com/github/'