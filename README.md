# dimm-error-pattern-visualization

## Data requirements
In order for the code to work properly, the input data has to fulfill some conditions. 
The input .csv has to contain the following columns:

- datetime: timestamp when the datapoint was collected

To identify the memory module uniquely:
- socket: number of the socket the investigated DIMM module belongs to
- imc: number of the integrated memroy controller the investigated DIMM module belongs to
- channel: number of channel the investigated DIMM module belongs to
- dimm: number of the respective DIMM the error occured on

To identify cells of the DIMM module uniquely:
- rank: number of the rank on the DIMM the error occured on
- bank: number of the bank on the rank the error occured on
- column: the column on the bank where the error occured
- row: the row on the bank where the error occured



## Dependencies
Install dependencies via 
```
pip install -r .\requirements.txt
```

## Parameters and Constants

The function *vis_single_dimm* requires two arguments:
- path: the path to the file containing our error data
- dimm: the id of the dimm we want to visualize

In addition to the function parameters we have to declare some constants depending on the hardware the data is taken from:
- RANKS: is the max number of ranks in our architecture
- BANKS: is the max number of banks per rank
- ROWS: is the max number of rows per bank
- COLUMNS: is the max number of columns per bank
