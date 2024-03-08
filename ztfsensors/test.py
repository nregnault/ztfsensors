
def get_pocket_test(filename = "ztf_20200401152477_000517_zg_c06_o.fits.fz"):
    """ get the pocket model and a raw-data_and_overscan test case.
    
    Returns
    -------
    PocketModel, 2d-array
        - pockelmodel
        - pixel

    Example
    -------
    pocket_model, pixels = get_pocket_test()
    %time _ = pocket_model.apply(pixels, backend="numpy")

    """
    import ztfimg
    import numpy as np
    from ztfsensors import pocket
    
    
    # Access the raw quadrant    
    rawimg = ztfimg.RawCCD.from_filename(filename, as_path=False) # providing the exact path
    quad = rawimg.get_quadrant(1)
    
    # Get data with overscan at the end
    data_and_overscan = quad.get_data_and_overscan()
    
    # the model with quadrant's pocket parameter
    pocket_model = pocket.PocketModel(**pocket.get_config(quad.ccdid, quad.qid).values[0])
    
    current_state = data_and_overscan.copy()
    current_state[:,-30:] = 0. # set overscan to zero
    current_state[0:2] = np.median(data_and_overscan) # default
    
    return pocket_model, current_state
