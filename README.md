# mnist-ocr

Recognize handwritten digits

## Tools Used

This project is to demonstrate the use of various tools:

- [cookiecutter data science project template](https://drivendata.github.io/cookiecutter-data-science/)
- Tensorflow
- GitHub Actions
- DVC for tracking large data and pipelines
<!-- - DagsHub -->
<!-- - MLFlow for tracking model parameters, metrics and model registry -->
- CML for generating reports with metrics and plots in each Git Pull Request
- Streamlit for Web UI

## Manual deployment on Azure

```bash
# add deploy key if repo is private
sudo apt install python3-pip
sudo pip install poetry
git clone git@github.com:blesswinsamuel/mnist-ocr.git
cd mnist-ocr
poetry install
poetry run streamlit run src/visualization/visualize.py # in tmux
```
