#!/usr/bin/env python 

import logging
import ztfsensors
from ztfsensors import pocket, correct 
import ztfimg


if __name__ == '__main__':
    exp_name = "ztf_20200401152477_000517_zg_c06_o.fits.fz"
    qid = 3
    
    logging.info(f'handle on: {exp_name}')
    rawimg = ztfimg.RawCCD.from_filename(exp_name, 
                                         as_path=False)
    
    logging.info(f'loading quadrant {qid}')
    quad = rawimg.get_quadrant(qid)
    pixels = quad.get_data_and_overscan()
    print(pixels.shape)
    
    config = pocket.get_config(quad.ccdid, quad.qid).iloc[0]
    model = pocket.PocketModel(**config)
    # corrected_pixels_and_overscan = correct.correct_pixels(model, pixels, n_overscan=30)



