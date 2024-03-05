from ruamel.yaml import YAML

def get_coef_model(idx_ccd, idx_quad, mjd,f_cor="pocket_corrections.yaml"):
    """
    alpha, cmax, beta, nmax = get_coef_model(idx_ccd, idx_quad, mjd)
    """
    data = open(f_cor)
    coefs = data.read()
    data.close()
    #
    yaml=YAML(typ='safe') 
    d_coefs =yaml.load(coefs)
    for coefs in d_coefs["data"]:
        #print(coefs["ccdid"], coefs["qid"])
        # TODO : manage mjd
        if coefs["ccdid"] == idx_ccd and coefs["qid"] == idx_quad:
            pars = coefs["pars"]
            print(pars)
            return pars["alpha"], pars["cmax"], pars["beta"], pars["nmax"]
    return None, None, None, None


if __name__ == '__main__':
    get_coef_model(11,1,0)
        