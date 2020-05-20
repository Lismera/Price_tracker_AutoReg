# Petrol price prediction model

The model developed can predict the next week's petrol price (based on weekly information gathered preciously by ONS). 

There is a feature to add the actual observation in order to improve the quality of the model. Because of this, the repository will be updated weekly. The new recordings is added to the "database" (csv file for now) automatically, no further interaction is needed to re-train the model. 


## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install all the dependencies in requirements.txt file on your virtualenv. Don't forget to git clone the repository first and cd into the folder cloned.

```bash
pip install -r requirements.txt
```

## Usage

If you run the code, in the command line it will print the predictions for next week price. 

Uncomment some parts of the code if you want to see:

* the initial data sample graph
* the predicted data (green) vs the original data stored in the .csv file (blue)


## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.


## License
Developed by Anastasia Ugaste. Modify and use as you wish.
