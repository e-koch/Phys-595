
'''
Bulk spectral line fitting with SDSS galaxy spectra
'''

import os
import shutil
from astropy.io import fits
from pandas import DataFrame, Series
import numpy as np
from multiprocessing import Pool
from datetime import datetime

# Bring in the package funcs
from specfit import do_specfit
from download_spectra import download_spectra


def bulk_fit(obs_file, output_file, keep_spectra=True, split_save=True,
             num_save=10, num_start=-1):
    '''
    Downloads files based off of the entries in the given file, performs
    spectral line fitting and saves the results to a FITS table.

    Parameters
    ----------

    num_start : int {<num_save}
        Used for restarting at a point set by num_save. Default is -1, which
        starts at the beginning.
    '''

    # Open the file
    data_file = fits.open(obs_file)

    spectra_data = data_file[1].data
    del data_file

    num_spectra = spectra_data.size
    save_nums = [(num_spectra/num_save)*(i+1) for i in range(num_save)]
    save_nums[-1] = num_spectra-1
    save_nums.append(0)

    start_pt = save_nums[num_start]

    for i in range(start_pt, num_spectra):
        spec_info = spectra_data[i]

        # Download the spectrum
        spec_name = \
            download_spectra(spec_info['PLATE'], spec_info['FIBERID'],
                             spec_info['MJD'], spec_info['SURVEY'])

        spec_name = "spectra/" + spec_name

        try:
            spec_df = do_specfit(spec_name, verbose=False)
        except IOError:
            download_spectra(spec_info['PLATE'], spec_info['FIBERID'],
                             spec_info['MJD'], spec_info['SURVEY'],
                             download=True)
            try:
                shutil.move(spec_name.split("/")[-1], spec_name.split("/")[0])
                spec_df = do_specfit(spec_name, verbose=False)
            except IOError:
                print("Could not download requested file.")
                # Put in the default size (11 lines fitted).
                spec_df = Series([np.NaN] * 44)
        if i == start_pt:
            df = DataFrame(spec_df, columns=[spec_name[:-5]])
        else:
            df[spec_name[:-5]] = spec_df

        if split_save and i in save_nums:
            posn = [j for j, x in enumerate(save_nums) if x == i][0]
            df.to_csv(output_file[:-4]+"_"+str(posn+1)+".csv")
        if not keep_spectra:
            os.system('rm ' + spec_name)

    if not split_save:
        df.to_csv(output_file)

    return


def parallel_bulkfit(path, num_splits=10, ncores=8, start_pt=0):
    '''
    Run bulk fitting in parallel. Results are outputted in chunks to make
    restarting easier.
    '''

    spectra = [f for f in os.listdir(path) if f[-4:] == 'fits']

    split_at = len(spectra) / num_splits

    splits = [split_at*i for i in range(1, num_splits)]
    splits.append(len(spectra))

    splits = splits[start_pt:]

    print splits

    prev_split = 0

    for i, split in enumerate(splits):

        print("On split " + str(i+1) + " of " + str(len(splits)))
        print(str(datetime.now()))

        split_spectra = spectra[prev_split:split]

        print len(split_spectra)

        # pool = Pool(processes=ncores)

        # output = pool.map(do_specfit, split_spectra)

        # pool.join()
        # pool.close()

        # df = DataFrame(output[0], columns=split_spectra[0])

        # for out, spec in zip(output[1:], split_spectra[1:]):
        #     df[spec[:-5]] = out

        # df.to_csv("spectral_fitting_"+str(i+1)+".csv")

        prev_split = split

if __name__ == "__main__":

    import sys

    input_file = str(sys.argv[1])
    output_file = str(sys.argv[2])
    restart_point = int(sys.argv[3])

    parallel = sys.argv[4]

    if parallel == 'True':
        parallel = True

    if parallel:
        ncores = int(sys.argv[5])
        parallel_bulkfit(input_file, start_pt=restart_point, ncores=ncores)
    else:
        bulk_fit(input_file, output_file, num_start=restart_point)
