# DS-ML-PRACTICE
Practice with Perfection.


Run these from your project terminal.

cd /home/jairajsingh/DS-ML-PRACTICE
Create and activate a local virtual environment:

python3 -m venv .venv
source .venv/bin/activate
Upgrade pip and install Jupyter kernel support:

python -m pip install --upgrade pip
python -m pip install ipykernel jupyter pandas numpy matplotlib seaborn scikit-learn
Register the environment as a notebook kernel:

python -m ipykernel install --user --name ds-ml-practice --display-name "Python (.venv DS-ML-PRACTICE)"
Then in Cursor/Jupyter:

Open house_price_prediction_Regression/notebooks/01_data_loading.ipynb
Click the kernel selector in the top-right of the notebook
Choose Python (.venv DS-ML-PRACTICE)
Run the cells again
If you only want the quickest fix without creating a virtual environment, run:

/bin/python3 -m pip install ipykernel -U --user --force-reinstall
But the .venv approach is cleaner for this project.