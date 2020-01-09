# Please download the codes and place them in the path ".\spinup\algos\BAIL_progressive" under your spinningup library 
The main function bail_learn() is in main_prog_bail.py. Run Progressive BAIL with it. 

Configuring the first parameter algo = 'bail_1_buf' or algo = 'bail_2_bah' determines which Progressive BAIL implementation you use.

I provide a config.json file to indicate best hyperparameters for Progressive BAIL, this is likely updated. Note those that are not hyperparameters："exp_name", "logger_kwargs", "seed" are specific experiment info; "buffer_type" determines the batch RL data used.

# I shall provide some datasets for your test runs soon

You may use Pytorch >= 1.1.0 for reproducing results of this code.
