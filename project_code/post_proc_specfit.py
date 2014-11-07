
'''
Post-process spectral line fitting results
'''

import numpy as np
from pandas import read_csv


def blank_the_crap(filename, min_amp_sn=3, min_wid_sn=3):
    '''
    Function to blank bad fits. Since the fits performed are highly
    restricted, bad fits are assumed to be non-detections.

    Along with the minimum S/N inputs, any error less than 0 is blanked.
    '''

    data = read_csv(filename)


    data_copy = data.T.copy()
    # There are 11 fitted lines and 4 parameters

    for i in range(11):
        line_pars = \
            [data.iloc[i][1:],
             data.iloc[i+11][1:],
             data.iloc[i+22][1:],
             data.iloc[i+33][1:]]

        good_err = np.logical_and(line_pars[1] > 0,
                             line_pars[3] > 0)

        good_err = np.append([1], good_err)

        good_sn_amp = (np.abs(line_pars[0]/line_pars[2]) >= 3)
        good_sn_amp = np.append([1], good_sn_amp)

        good_sn_wid = (np.abs(line_pars[1]/line_pars[3]) >= 3)
        good_sn_wid = np.append([1], good_sn_wid)


        data_copy[i] = data_copy[i][(good_err + good_sn_amp + good_sn_wid) == 1]
        data_copy[i+11] = data_copy[i+11][(good_err + good_sn_amp + good_sn_wid) == 1]
        data_copy[i+22] = data_copy[i+22][(good_err + good_sn_amp + good_sn_wid) == 1]
        data_copy[i+33] = data_copy[i+33][(good_err + good_sn_amp + good_sn_wid) == 1]


    data_copy.to_csv(filename[:-4] + "_cleaned.csv")