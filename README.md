# Conversational Chatbot

## Description
This project provides a detailed guide on setting up a Python virtual environment, installing necessary requirements, and running a Jupyter Notebook with a custom kernel.

## Prerequisites
- Python 3.10 must be installed on your system.

## Setup Instructions

### 1. Create and Activate Virtual Environment
First, create a virtual environment. This helps to manage dependencies and avoid conflicts with other projects.

```bash
python -m venv .venv
```

Activate the virtual environment:

- On Windows:
  ```bash
  .\.venv\Scripts\activate
  ```
- On Unix or MacOS:
  ```bash
  source .venv/bin/activate
  ```

### 2. Install Requirements
Once the virtual environment is activated, install the necessary requirements using `pip`.

```bash
pip install -r requirements.txt
```

### 3. Install IPython Kernel
To use your virtual environment with Jupyter Notebook, you need to install the IPython kernel.

```bash
python -m ipykernel install --user --name=.venv --display-name "Python (myenv)"
```

### 4. Start Jupyter Notebook
Finally, start the Jupyter Notebook.

```bash
jupyter notebook conversational_chatbot.ipynb
```