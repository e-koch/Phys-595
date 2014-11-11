
'''
Post-process spectral line fitting results
'''

import numpy as np
from pandas import read_csv, notnull, Series, concat


def concat_csvs(file_list, output_name, save=True):
    '''
    Concatenate csv files.
    '''

    data = read_csv(file_list[0])

    for name in file_list[1:]:
        data = concat(data, read_csv(name))

    data.index = data['Unnamed: 0']

    del data['Unnamed: 0']

    if save:
        data.to_csv(output_name)
    else:
        return data


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
            [np.asarray(data.iloc[i][1:]),
             np.asarray(data.iloc[i+11][1:]),
             np.asarray(data.iloc[i+22][1:]),
             np.asarray(data.iloc[i+33][1:])]

        good_err = np.logical_or(line_pars[2] > 0,
                                 line_pars[3] > 0)

        posns_err = np.where(good_err == 0)

        good_sn_amp = \
            np.logical_or(np.abs(line_pars[0]/line_pars[2]) >= min_amp_sn,
                          notnull(line_pars[0]/line_pars[2]))

        posns_amp = np.where(good_sn_amp == 0)

        good_sn_wid = \
            np.logical_or(np.abs(line_pars[1]/line_pars[3]) >= min_wid_sn,
                          notnull(line_pars[1]/line_pars[3]))

        posns_wid = np.where(good_sn_wid == 0)

        line_pars[0][posns_err[0]] = 0.0
        line_pars[1][posns_err[0]] = 0.0
        line_pars[2][posns_err[0]] = 0.0
        line_pars[3][posns_err[0]] = 0.0

        line_pars[0][posns_amp[0]] = 0.0
        line_pars[1][posns_amp[0]] = 0.0
        line_pars[2][posns_amp[0]] = 0.0
        line_pars[3][posns_amp[0]] = 0.0

        line_pars[0][posns_wid[0]] = 0.0
        line_pars[1][posns_wid[0]] = 0.0
        line_pars[2][posns_wid[0]] = 0.0
        line_pars[3][posns_wid[0]] = 0.0

        data_copy[i] = Series(line_pars[0], index=data_copy.index)
        data_copy[i+11] = Series(line_pars[1], index=data_copy.index)
        data_copy[i+22] = Series(line_pars[2], index=data_copy.index)
        data_copy[i+33] = Series(line_pars[3], index=data_copy.index)


        line_pars[0]["Unnamed: 0"] = data_copy[i][0]
        line_pars[1]["Unnamed: 0"] = data_copy[i+11][0]
        line_pars[2]["Unnamed: 0"] = data_copy[i+22][0]
        line_pars[3]["Unnamed: 0"] = data_copy[i+33][0]

    print ah

    data_copy.to_csv(filename[:-4] + "_cleaned.csv")
