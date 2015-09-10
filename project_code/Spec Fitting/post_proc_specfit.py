
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
                   num_lines=12, remove_failed=True):
    '''
    Function to blank bad fits. Since the fits performed are highly
    restricted, bad fits are assumed to be non-detections.

    Along with the minimum S/N inputs, any error less than 0 is blanked.

    Parameters
    ----------
    remove_failed : bool, optional
        Uses spurious_blanking to remove small number of
        failed fits.

    '''

    data = read_csv(filename, index_col=0)

    df = data.copy()

    df = spurious_blanking(df)

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


def spurious_blanking(df):
    '''
    A small number of spectra have various systematic problems.
    Most just show up as terrible fits, and are removed, but a few
    completely fail.
    '''

    bad_halp_nii = \
        [u'spec-0931-52619-0176', u'spec-1240-52734-0249',
         u'spec-2153-54212-0176', u'spec-2478-54097-0218',
         u'spec-2003-53442-0250', u'spec-2286-53700-0250']

    for spec in bad_halp_nii:
        df.ix[spec]['NIIa Amplitude'] = 0.0
        df.ix[spec]['Halp Amplitude'] = 0.0
        df.ix[spec]['NIIb Amplitude'] = 0.0

        df.ix[spec]['NIIa Width'] = 0.0
        df.ix[spec]['Halp Width'] = 0.0
        df.ix[spec]['NIIb Width'] = 0.0


    bad_ca_hk = \
        [u'spec-1085-52531-0278', u'spec-1053-52468-0523',
         u'spec-1320-52759-0280', u'spec-2642-54232-0522']

    for spec in bad_ca_hk:
        df.ix[spec]['Ca H Amplitude'] = 0.0
        df.ix[spec]['Ca K Amplitude'] = 0.0

        df.ix[spec]['Ca H Width'] = 0.0
        df.ix[spec]['Ca K Width'] = 0.0

    return df


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
