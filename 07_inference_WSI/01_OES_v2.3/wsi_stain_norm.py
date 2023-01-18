#FUNCTIONS FOR STAIN NORMALIZATION

import staintools

#Inititate BrightnessStandardizer
standardizer = staintools.BrightnessStandardizer()

#Inititate StainNormalizer "macenko"
stain_norm = staintools.StainNormalizer(method='macenko')