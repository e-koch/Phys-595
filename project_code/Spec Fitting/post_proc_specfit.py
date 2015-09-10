
'''
Post-process spectral line fitting results
'''

import numpy as np
from pandas import read_csv, Series, concat
import shutil


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


def blank_the_crap(filename, min_amp_sn=3, min_width_sn=3,
                   num_lines=12):
    '''
    Function to blank bad fits. Since the fits performed are highly
    restricted, bad fits are assumed to be non-detections.

    Along with the minimum S/N inputs, any error less than 0 is blanked.
    '''

    data = read_csv(filename, index_col=0)

    df = data.copy()

    # There are 12 fitted lines and 4 parameters

    columns = df.columns

    assert len(columns) == num_lines * 4

    # df["Halp Amplitude"][df.index[np.where((df['Halp Amplitude']/df["Halp Amplitude Error"]).abs() <=1)]] = 0.0

    for i in range(num_lines):

        amp = df[columns[i]]
        width = df[columns[i+num_lines]]
        amp_err = df[columns[i+2*num_lines]]
        width_err = df[columns[i+3*num_lines]]

        # Remove fits where either error is 0
        bad_errs = \
            df.index[np.where(np.logical_or(amp_err <= 0.0,
                                            width_err <= 0.0))]

        bad_sn_amp = df.index[np.where(np.abs(amp / amp_err) < min_amp_sn)]

        bad_sn_width = \
            df.index[np.where(np.abs(width / width_err) < min_width_sn)]

        amp[bad_errs] = 0.0
        amp[bad_sn_amp] = 0.0
        amp[bad_sn_width] = 0.0

        width[bad_errs] = 0.0
        width[bad_sn_amp] = 0.0
        width[bad_sn_width] = 0.0

        df[columns[i]] = amp
        df[columns[i+num_lines]] = width

    # Save cleaned version without the error columns

    df.iloc[:, :2*num_lines].to_csv(filename[:-4] + "_cleaned.csv")


def make_weighted_df(df):
    '''
    Weight by the inverse squared of the errors.
    '''
    pass

def collect_spectra(filename, path='anomalies/', verbose=True):
    '''
    Given a dataframe with an index of files, find those files and copy them
    to a new directory.
    '''

    df = read_csv(filename)

    # Spectra names
    names = df['Unnamed: 0.1'].drop_duplicates()

    # The files could be in any of 4 places
    prefixes = ["samples_1/", "samples_2/", "samples_3/", "samples_4/"]

    for name in names:
        # Need to make sure the filename is right...
        if len(name.split("-")[1]) < 4:
            add_zeros = 4 - len(name.split("-"))
            split_name = name.split("-")
            split_name[1] = "0"*add_zeros + split_name[1]
            name = "-".join(split_name)
        i = 0
        while True:
            try:
                shutil.copy("/mnt/"+prefixes[i]+name+".fits", "/mnt/"+path)
                i = 0
                break
            except IOError:
                if i > 3:
                    raise TypeError("Cannot find spectrum named: " + name)
                i += 1
