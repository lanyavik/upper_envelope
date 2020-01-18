# BAIL procedure

1. generate the datasets

2. run main_get_mcret.py to get the S.npy and Gain.npy files

3.(a)  run main_stat_bail.py 
       which trains an upper envelope, plot the envelope, selects a "best state-action" dataset and do behaviral cloning

3.(b)  run main_prog_bail.py (choose either "bail_1_buf" or "bail_2_bah" of the progressive bail implementation)
       which iteratively do one gradient update on upper envelope, select state-action in the dataset/mini-batch via upper envelope, 
       one gradient update on the actor network using behavioral cloning
