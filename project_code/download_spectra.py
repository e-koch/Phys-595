
'''
Download SDSS spectra from the plate ID and fibre
'''

import os


def download_spectra(plateid, fibreid, mjd, survey):
    '''
    Download an SDSS spectrum given the parameters.
    '''

    # Convert to lowercase
    plateid = plateid.lower()
    fibreid = fibreid.lower()
    survey = survey.lower()

    filename = plateid+"/spec-"+plateid+"-"+mjd+"-"+fibreid+".fits"

    if survey is 'boss':
        file_suffix = survey+"/spectro/redux/v5_5_12/spectra/"+filename
    elif survey is 'sdss_legacy':
        file_suffix = "sdss/spectro/redux/26/spectra/"+filename
    elif survey is 'sdss_stel_clust':
        file_suffix = "sdss/spectro/redux/103/spectra/"+filename
    elif survey is 'sdss_segue':
        file_suffix = "sdss/spectro/redux/104/spectra/"+filename
    else:
        raise NameError("Check the survey name. Inputted was %s" % (survey))

    # Check to make sure that file exists
    os.system("wget http://data.sdss3.org/sas/dr10/"+file_suffix)

    return filename


def split_spectra_list(props_file, n_split=4):
    '''
    Take the list of spectra in the given file and split it into equal
    portions.
    '''

    from astropy.io import fits

    data = fits.open(props_file)

    header = data[0].header
    data = data[1].data

    num_spec = data['Z'].shape[0]

    num_split = num_spec / n_split
    left_split = [1] * (num_spec % n_split) + \
                 [0] * (n_split - num_spec % n_split)

    for num, rem in zip(range(1, n_split + 1), left_split):
        part = data[(num-1) * num_split:num * num_split + rem]

        filename = props_file[:-5] + "_part" + str(num) + ".fits"

        hdu = fits.HDUList([fits.PrimaryHDU(header), part])

        hdu.writeto(filename)

    return
