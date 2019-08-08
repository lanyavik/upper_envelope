from spinup.utils.run_utils import ExperimentGrid
from spinup.algos.BCQ.main_bc_ue_border import bc_ue_learn
import time
import gym


rollout_list = [None, 1000, 200, 100, 10]
k_list = [10000, 1000, 100]

if __name__ == '__main__':
    import argparse
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', type=int, default=1)
    parser.add_argument('--setting', type=int, default=0)
    args = parser.parse_args()
    ## MAKE SURE ALPHA IS ADDED, MAKE SURE EACH SETTING IS ADDED
    ## MAKE SURE exp name is change, make sure used correct sac function
    exp_name = 'BC_ue'
    setting_names = ['env_set', 'seed', 'border', 'lr', 'buffer_seed', 'ue_seed']
    settings = [['HalfCheetah-v2'],
               [1, 2, 3, 4, 5], [0.85, 0.9, 0.92, 0.95], [0.001], [0, 1], [1, 11]]

##########################################DON'T NEED TO MODIFY#######################################
    ## this block will assign a certain set of setting to a "--setting" number
    ## basically, maps a set of settings to a hpc job array id
    total = 1
    for sett in settings:
        total *= len(sett)

    print("total: ", total)

    def get_setting(setting_number, total, settings, setting_names):
        indexes = []  ## this says which hyperparameter we use
        remainder = setting_number
        for setting in settings:
            division = int(total / len(setting))
            index = int(remainder / division)
            remainder = remainder % division
            indexes.append(index)
            total = division
        actual_setting = {}
        for j in range(len(indexes)):
            actual_setting[setting_names[j]] = settings[j][indexes[j]]
        return indexes, actual_setting

    indexes, actual_setting = get_setting(args.setting, total, settings, setting_names)
####################################################################################################

    ## use eg.add to add parameters in the settings or add parameters that apply to all jobs
    eg = ExperimentGrid(name=exp_name)
    eg.add('ue_seed', actual_setting['ue_seed'], 'ues', True)
    eg.add('lr', actual_setting['lr'], 'lr', True)
    eg.add('border', actual_setting['border'], 'border', True)
    eg.add('wd', 0, 'wd', True)
    eg.add('buffer_type', 'FinalSigma0.5', 'Buf-', True)
    eg.add('buffer_size', '1000K')
    eg.add('cut_buffer_size', '1000K', '', True)
    eg.add('buffer_seed', actual_setting['buffer_seed'], '', True)
    eg.add('eval_freq', 1500)
    eg.add('max_timesteps', 300000)
    eg.add('env_set', actual_setting['env_set'], '', True)
    eg.add('seed', actual_setting['seed'])

    eg.run(bc_ue_learn, num_cpu=args.cpu)

    print('\n###################################### GRID EXP END ######################################')
    print('total time for grid experiment:',time.time()-start_time)
