## Source Code for COMP5212 Programming Assignment 1

To run this code, you will need PyTorch 1.12, which is the latest version. GPU is optional and by default not used.

The easiest way is to directly run: 

```bash
python main.py
```

I set some predefined parameters. And this command will train a Logistic Regression model with SGD optimiser. 



To generate data need for report, open the following file

```bash
report_data.ipynb
```

Go through every line. Results and figures will be saved to `report` folder. Compile the latex file there and you will receive the report.



### Substructures

The model was defined under `models.py` file. And training script is in `main.py` file. 