
'''
Figure of lines to fit for proposal
'''

import matplotlib.pyplot as p
from astropy.io import fits
from scipy.ndimage import gaussian_filter1d

lines = [r"H$\alpha$-6562$\AA$ \& NII-6583$\AA$", r"H$\beta$", r"H$\gamma", r"H$\delta$",
         "Ca H & K", "MgII", "NaI", "OIIIa \& b"]
lambdas = [6562, 4861, 4340, 4103, 3950, 5175, 5894, 4959]

filename = "/Users/eric/../../Volumes/Mac_Storage/sdss_datamining/spec-0266-51602-0001.fits"

spec_file = fits.open(filename)

flux = spec_file[1].data["flux"]
smooth = gaussian_filter1d(flux, 2)
lam_wav = 10**spec_file[1].data["loglam"] / (1 + spec_file[2].data["Z"])


p.plot(lam_wav, smooth, 'b')
p.xlabel(r"Wavelength ($\AA$)")
p.ylabel(r"Flux ($10^{-17} erg/s/cm^2/\AA$)")
p.ylim(smooth.min(), smooth.max()+10)
p.xlim(lam_wav.min(), 6800)

for name, lam in zip(lines, lambdas):
    p.axvline(x=lam, color='k', linestyle='--')
    # p.annotate(name, xy=(lam, 60), xytext=(lam, 60))

p.annotate(r"H$\alpha$ - 6562$\AA$ \& NII - 6583$\AA$",
           xy=(6562, 50), xytext=(6562, 50), rotation=90,
           horizontalalignment='right',
           verticalalignment='center')

p.annotate(r"H$\beta$ - 4861$\AA$",
           xy=(4861, 110), xytext=(4861+5, 110), rotation=90,
           horizontalalignment='right',
           verticalalignment='center')

p.annotate(r"H$\gamma$ - 4340$\AA$",
           xy=(4340, 110), xytext=(4340+20, 110), rotation=90,
           horizontalalignment='left',
           verticalalignment='center')

p.annotate(r"H$\delta$ - 4103$\AA$",
           xy=(4103, 90), xytext=(4103+20, 90), rotation=90,
           horizontalalignment='left',
           verticalalignment='center')

p.annotate(r"Ca H \& K - 3934, 3969$\AA$",
           xy=(3950, 90), xytext=(3950, 90), rotation=90,
           horizontalalignment='right',
           verticalalignment='center')

p.annotate(r"MgII - 5175$\AA$",
           xy=(5175, 110), xytext=(5175+20, 110), rotation=90,
           horizontalalignment='left',
           verticalalignment='center')

p.annotate(r"NaI - 5894$\AA$",
           xy=(5894, 60), xytext=(5894, 60), rotation=90,
           horizontalalignment='right',
           verticalalignment='center')

p.annotate(r"[OIII] - 4959, 5007$\AA$",
           xy=(4959, 50), xytext=(4959+20, 45), rotation=90,
           horizontalalignment='left',
           verticalalignment='center')

p.show()

p.close()

# Two

filename = "/Users/eric/../../Volumes/Mac_Storage/sdss_datamining/spec-0273-51957-0136.fits"

spec_file = fits.open(filename)

flux = spec_file[1].data["flux"]
smooth = gaussian_filter1d(flux, 2)
lam_wav = 10**spec_file[1].data["loglam"] / (1 + spec_file[2].data["Z"])


p.plot(lam_wav, smooth, 'b')
p.xlabel(r"Wavelength ($\AA$)")
p.ylabel(r"Flux ($10^{-17} erg/s/cm^2/\AA$)")
p.ylim(smooth.min(), smooth.max()+5)
p.xlim(lam_wav.min(), 6800)

for name, lam in zip(lines, lambdas):
    p.axvline(x=lam, color='k', linestyle='--')
    # p.annotate(name, xy=(lam, 60), xytext=(lam, 60))

p.annotate(r"H$\alpha$ - 6562$\AA$ \& NII - 6583$\AA$",
           xy=(6562, 17), xytext=(6562-35, 15), rotation=90,
           horizontalalignment='right',
           verticalalignment='center')

p.annotate(r"H$\beta$ - 4861$\AA$",
           xy=(4861, 15), xytext=(4861+5, 15), rotation=90,
           horizontalalignment='right',
           verticalalignment='center')

p.annotate(r"H$\gamma$ - 4340$\AA$",
           xy=(4340, 15), xytext=(4340+20, 15), rotation=90,
           horizontalalignment='left',
           verticalalignment='center')

p.annotate(r"H$\delta$ - 4103$\AA$",
           xy=(4103, 15), xytext=(4103+20, 15), rotation=90,
           horizontalalignment='left',
           verticalalignment='center')

p.annotate(r"Ca H \& K - 3934, 3969$\AA$",
           xy=(3950, 15), xytext=(3950, 15), rotation=90,
           horizontalalignment='right',
           verticalalignment='center')

p.annotate(r"MgII - 5175$\AA$",
           xy=(5175, 15), xytext=(5175+20, 15), rotation=90,
           horizontalalignment='left',
           verticalalignment='center')

p.annotate(r"NaI - 5894$\AA$",
           xy=(5894, 15), xytext=(5894, 15), rotation=90,
           horizontalalignment='right',
           verticalalignment='center')

p.annotate(r"[OIII] - 4959, 5007$\AA$",
           xy=(4959, 15), xytext=(4959+20, 15), rotation=90,
           horizontalalignment='left',
           verticalalignment='center')

p.show()
