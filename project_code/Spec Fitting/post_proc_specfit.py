
'''
Post-process spectral line fitting results
'''

import numpy as np
from pandas import read_csv, Series, concat
import os


def concat_csvs(file_list, output_name, save=True):
    '''
    Concatenate csv files.
    '''

    data = [read_csv(file_nm) for file_nm in file_list]

    index = data[0]["Unnamed: 0"]

    for dat in data:
        del dat['Unnamed: 0']

    data = concat(data, ignore_index=False, axis=1)

    data.index = index

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
    del data["Unnamed: 0"]

    data_copy = data.T.copy()
    # There are 11 fitted lines and 4 parameters

    for i in range(11):
        line_pars = \
            [np.asarray(data.iloc[i]),
             np.asarray(data.iloc[i+11]),
             np.asarray(data.iloc[i+22]),
             np.asarray(data.iloc[i+33])]

        good_err_1 = (line_pars[2] > 0)
        good_err_2 = (line_pars[3] > 0)

        posns_err_1 = np.where(good_err_1 == 0)
        posns_err_2 = np.where(good_err_2 == 0)

        good_sn_amp = (np.abs(line_pars[0]/line_pars[2]) >= min_amp_sn)

        posns_amp = np.where(good_sn_amp == 0)

        good_sn_wid = (np.abs(line_pars[1]/line_pars[3]) >= min_wid_sn)

        posns_wid = np.where(good_sn_wid == 0)

        line_pars[0][posns_err_1[0]] = 0.0
        line_pars[1][posns_err_1[0]] = 0.0
        line_pars[2][posns_err_1[0]] = 0.0
        line_pars[3][posns_err_1[0]] = 0.0
        line_pars[0][posns_err_2[0]] = 0.0
        line_pars[1][posns_err_2[0]] = 0.0
        line_pars[2][posns_err_2[0]] = 0.0
        line_pars[3][posns_err_2[0]] = 0.0

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

    data_copy.to_csv(filename[:-4] + "_cleaned.csv")


def collect_spectra(filename, path='anomalies/'):
    '''
    Given a dataframe with an index of files, find those files and copy them
    to a new directory.
    '''

    df = read_csv(filename)

    # Spectra names
    names = df['Unnamed: 0'].drop_duplicates()

    # The files could be in any of 4 places
    prefixes = ["samples1/", "samples2/", "samples3/", "samples4/"]

    for name in names:
        i = 0
        while True:
            try:
                os.system("cp "+prefixes[i]+name+" "+path)
                i = 0
                break
            except OSError:
                if i > 3:
                    raise TypeError("Cannot find spectrum named: " + name)
                i += 1
