# ReadMe file - src folder

# Introduction
The 'src' folder is a result of portfolio assignment #4.
This folder contains the hypertune.py file which was used to experiment with Ray for setting hyperparameters.

This set-up was done based on the 'Make a module' tutorial. This was the first time i had learned about setting up an application or module this way.

Because the Surf VM had issues with memory usage for a second .venv, the uv-lock and pyproject.toml of the 'Portfolio-S3' folder was used. Ideally, the src-folder would be a separated folder with it's own dependencies.
In the current situation it is a part of the portfolio-S3 folder and therefore not sharable as a module or application.

Alltough this is not a production-like set-up, following the tutorial was still a very useful experience. 

** Update: ** After the deployment assignment it became a lot more clear how to deal with different .venv's and packages while working on a VM. With this new information i would not have set up the week 4 package like this. Due to time concerns i did not change the current set-up, but is was a nice learning experience tough. 

# Structure

```
portfolio-S3/         # Project root
|- .venv/             # Virtual environment directory
├── src/ 
│ └── week4_hypertune/ # Module for week 4 hyperparameter tuning 
│   ├── __init__.py 
│   ├── hypertune.py 
│   └── README.md
|- pyproject.toml     # Project dependencies
|- uv.lock            # Project dependencies
|- data
|- logs
|- hypertune.log
| All other folders from portfolio-S3.

``` 

# Results

1.  **[Hyperparameter ray- PDF - Assignment 4](../../4-hypertuning-ray/DL-portfolioassignment-4-Ray-Jeffrey-Gerrits-15-03-2025.pdf)**
    * Portfolio assignment 4
