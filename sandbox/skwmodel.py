#!/usr/bin/env python

"""
Simulation of a test data sample
"""
import logging

import emcee
import numba
import numpy as np
import pandas
import pylab as pl
from astropy.time import Time
from croaks import DataProxy
from numba.experimental import jitclass
from saunerie.fitparameters import FitParameters
from scipy.optimize import least_squares

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', 
                    level=logging.INFO)

def sim(size=1000, skylev=100, flux=1000, x_star=500.2, sigma=4., noise=True, plot=False):
    """
    """
    x = np.arange(0, size, 1.)
    y = np.full(size, skylev)
    psf = np.exp(-0.5 * ((x-x_star)/sigma)**2) / np.sqrt(2 * np.pi) / sigma
    y = y + flux * psf
    if noise:
        y = y + np.random.normal(loc=0., scale=np.sqrt(y), size=size)
    
    y[-30:] = 0.
    c = np.zeros_like(y)
    s = np.zeros_like(y)
    from_pocket = np.zeros_like(y)
    to_pocket = np.zeros_like(y)
    delta = np.zeros_like(y)
    if plot:
        pl.plot(x, y, 'k.')

    return np.vstack((y, c, s, from_pocket, to_pocket, delta))

def fill(c, n, p):
    cmax, deltamax, alpha, n0 = p[0], p[1], p[2], p[3]
    nn = np.arctan(n/n0) * 2 / np.pi
    b = deltamax**(1./alpha)
    a = -b / cmax
    cc = np.power(a*c+b, alpha)
    return cc * nn

def flush(c, n, p):
    cmax, deltamax, alpha, n0 = p
    return deltamax * np.power(c/cmax, alpha)

def plot_fill_flush(n, p):
    pl.figure()
    cmax = p[0]
    c = np.linspace(0, cmax, 100)
    pl.plot(c, fill(c, n=n, p=p), 'b.-', label='fill')
    pl.plot(c, flush(c, n=n, p=p), 'r.-', label='flush')
    pl.legend(loc='best')


def add_fq(d, fill, flush, p):
    _,N = d.shape
    for i in range(1,N):
        n = d[0,i] # current charge, to be read
        c = d[1,i-1]
        d[3,i] = fill(c, n, p)
        d[4,i] = flush(c, n, p)# charge currently stored in the pocket
        delta_c =  fill(c, n, p) - flush(c, n, p)
        d[1,i] = d[1,i-1] + delta_c
        d[2,i] = d[0,i] - delta_c

def plot_evol(d):
    fig, ax = pl.subplots(nrows=4, ncols=1, sharex=True, figsize=(10,10))
    ax[0].plot(d[0,:], 'k.', label='$n_i$')
    ax[0].plot(d[2,:], 'b+', label='$s_i$ (measured)')
    ax[0].legend(loc='best')
    ax[0].set_ylabel('signal')
    ax[1].plot(d[1,:], 'r.')
    ax[1].set_ylabel('pocket contents')
    ax[2].plot(d[3,:], 'b.', label='fill')
    ax[2].plot(d[4,:], 'r+', label='flush')
    ax[2].legend(loc='best')
    ax[3].plot(d[3]-d[4,:], 'b.')
    ax[3].set_ylabel('net flux')

    pl.figure()
    pl.semilogy(d[2,-30:], 'k.')
    pl.ylabel('overscan')
    pl.xlabel('col')

def load(ccdid=6, qid=1, start_date='2019-12-01', filename='overscan.parquet'):
    """load the data, and build a data proxy 

    """
    d = pandas.read_parquet(filename)
    start_mjd = Time(start_date).mjd
    idx = (d.ccdid==ccdid) & (d.qid==qid) & (d.mjd>=start_mjd) & (d.quadrant.str[-9]=='o')
    dd = d[idx]
    data_proxy = DataProxy(dd, mjd='mjd', skylev='skylev', j='j', overscan='overscan', 
                           mean_trail_overscan='mean_trail_overscan', quadrant='quadrant', 
                           temp='temp', head_temp='head_temp', detheat='detheat', 
                           azimuth='azimuth', elevation='elevation') 
    data_proxy.add_field('y', dd.overscan - dd.mean_trail_overscan)
    data_proxy.make_index('quadrant')
    data_proxy.make_index('mjd')
    data_proxy.make_index('skylev')
    data_proxy.make_index('temp')
    data_proxy.make_index('head_temp')
    data_proxy.make_index('detheat')
    data_proxy.make_index('azimuth')
    data_proxy.make_index('elevation')
    data_proxy.add_field('lastcol', dd.lastcol - dd.mean_trail_overscan)
    data_proxy.make_index('lastcol')
    return data_proxy

class OverscanModel:

    def __init__(self, dp):
        """Constructor 
        
        Unfold the data proxy into a 2D matrix containing all the overscan data
        """
        self.dp = dp
        self.nquads = len(self.dp.quadrant_set)
        self.ncols = self.dp.j.max()+1
        self.overscan_data = dd = np.zeros((self.nquads, self.ncols))

        # if order_by_quadrant:
        # print('order by quadrant !')
        self.exp_index = self.dp.quadrant_index
        # else:
        #     print('order by mjd !')
        #     self.exp_index = self.dp.mjd_index
        dd[self.exp_index, self.dp.j] = self.dp.y
        self.pocket = np.zeros((self.nquads, self.ncols+1))
        self.pars = self.init_pars()

    def init_pars(self):
        """initialize and return a parameter vector
        """
        pars = FitParameters([('cmax', 1), ('deltamax', 1), ('alpha', 1), ('n0', 1), ('pocket', self.nquads)])
        pars.full[0:4] = [3300., 5000., 1.7, 300.]
        pars.full[4:] = 1000.
        return pars

    def _fill(self, c, n):
        """fill function: how many electrons may escape into the pocket

        Parameters:
        -----------
        c : (float) 
          pocket charge
        s : (float)
          pixel charge
        """
        pocket_max_charge, max_transfer_rate, alpha, n0 = self.pars.full[:4]
        nn = np.arctan(n/n0) * 2 / np.pi
        b = max_transfer_rate ** (1./alpha)
        a = -b/pocket_max_charge
        cc = np.power(a*c+b, alpha)
        return cc * nn

    def _flush(self, c):
        """flush function: how many electrons may come back from the pocket

        Parameters:
        -----------
         c : (float)
           pocket charge
        """
        pocket_max_charge, max_transfer_rate, alpha, n0 = self.pars.full[:4]
        flsh = max_transfer_rate * np.power(c/pocket_max_charge, alpha)
        return flsh

    def __call__(self, free_pars, debug=False): 
        """evaluate the model and return residuals
        """
        self.pars.free = free_pars
        n_electrons = np.zeros(self.nquads)
        pocket = self.pocket
        pocket[:,:] = 0.
        pocket[:,0] = self.pars['pocket'].full[:]
        if debug:
            to_pckt = np.zeros((self.nquads, self.ncols+1))
            from_pckt = np.zeros((self.nquads, self.ncols+1))
        signal = np.zeros((self.nquads, self.ncols+1))

        for i in range(1, self.ncols+1):
            pocket_charge_before_readout = pocket[:,i-1]
            to_pocket = self._fill(pocket_charge_before_readout, n_electrons)
            from_pocket = self._flush(pocket_charge_before_readout)
            signal[:,i] = - to_pocket + from_pocket
            pocket[:,i] = pocket_charge_before_readout + to_pocket - from_pocket
            if debug:
                to_pckt[:,i] = to_pocket
                from_pckt[:,i] = from_pocket

        if debug: 
            return signal, pocket, from_pckt, to_pckt

        return (self.overscan_data - signal[:,1:]).ravel()

class OverscanModelNoDeltaMax(OverscanModel):

    def __init__(self, dp): 
        super(OverscanModelNoDeltaMax, self).__init__(dp)

    def init_pars(self):
        pars = FitParameters([('cmax', 1), ('alpha', 1), ('n0', 1), ('beta', 1), ('pocket', self.nquads)])
        pars.full[0:4] = [3300., 1.7, 600., 1.]
        pars.full[4:] = 1000.
        return pars

    def _fill(self, c, n):
        """pocket fill function. Same version as before. With no max transfer rate.
        """
        c, n = np.array(c), np.array(n)
        pocket_max_charge, alpha, n0, beta = self.pars.full[:4]
        x = c / pocket_max_charge
        # nn = beta * (np.arctan(n / n0) * 2 / np.pi)
        nn = np.power(n/n0, beta)
        to_pocket = pocket_max_charge * nn * np.power(1-x, alpha)
        if hasattr(self, 'debug'):
            print('  --> DEBUG: ', pocket_max_charge, n, n0, n/n0, np.power(1-x, alpha)) # print('_fill: ', pocket_max_charge, alpha, n0, ' | ', x, nn, to_pocket)
        return to_pocket

    def _d_fill(self, c, n):
        """
        """
        pocket_max_charge, alpha, n0, beta = self.pars.full[:4]
        x = c / pocket_max_charge
        nn = np.power(n/n0, beta)
        dnn = beta * np.power(n/n0, beta-1.) / n0
        pp = np.power(1-x, alpha)
        dpp = -alpha * np.power(1-x, alpha-1.)
        dvaldn = pocket_max_charge * dnn*pp
        dvaldc = nn*dpp
        return np.array([dvaldc, dvaldn]).T

    def _flush(self, c):
        """
        """
        pocket_max_charge, alpha, n0, beta = self.pars.full[:4]
        x = c/pocket_max_charge
        from_pocket = pocket_max_charge * np.power(x, alpha)
        # print('_flush: ', pocket_max_charge, alpha, n0, ' | ', x, from_pocket)
        return from_pocket

    def _d_flush(self, c):
        """
        """
        c = np.array(c)
        pocket_max_charge, alpha, n0, beta = self.pars.full[:4]
        x = c/pocket_max_charge
        # dval = np.array(pocket_max_charge * alpha * np.power(x, alpha-1))
        dval = np.array(alpha * np.power(x, alpha-1))
        z = np.zeros_like(dval)
        return np.array([dval, z]).T

    def plot_fill_flush(self):
        """
        """
        pocket_max_charge, alpha, n0, beta = self.pars.full[:4]
        pocket_charge = np.linspace(0, pocket_max_charge, 100)
        pl.figure(figsize=(8,8))
        pl.plot(pocket_charge, self._flush(pocket_charge), 'k-', label='from_pocket')
        for pixel_charge in (200., 500., 1000., 5000.):
            pl.plot(pocket_charge, self._fill(pocket_charge, pixel_charge), color='r', ls='-', 
                    label=f'to_pocket [$n_{{e^-}}^{{pix}}={pixel_charge}$]')
            pcc = self.pocket_charge(pixel_charge)
            pl.axvline(pcc, ls=':')
            from_pocket = self._flush(pcc)
            to_pocket = self._fill(pcc, pixel_charge)
            pl.plot([pcc], [from_pocket], 'ko')
            pl.plot([pcc], [to_pocket], 'r*')
        pl.xlabel('pocket charge')
        pl.ylabel('transfer rate')
        pl.legend(loc='best')
        pl.title('Fill and Flush functions')
        pl.text(0.32, 0.70, f'$\\alpha={alpha:.3f}$', transform=pl.gca().transAxes)
        pl.text(0.32, 0.65, f'pocket size={pocket_max_charge:.1f}', transform=pl.gca().transAxes)

    def pocket_charge(self, neq):
        """
        """
        pocket_max_charge, alpha, n0, beta = self.pars.full[:4]
        # g_eq = np.power(beta * np.arctan(neq/n0) * 2. / np.pi, 1./alpha)
        g_eq = np.power(neq/n0, beta/alpha)
        return pocket_max_charge * g_eq / (1. + g_eq)

    def apply(self, d):
        """Apply the model to time series of simulated data
        """
        _,N = d.shape
        for i in range(1,N):
            n_i = d[0,i]
            q_ii = d[1,i-1]
            from_pocket = self._flush(q_ii)
            to_pocket = self._fill(q_ii, n_i)
            # print(n_i, q_ii, from_pocket, to_pocket)
            delta = from_pocket - to_pocket
            d[2,i] = n_i + delta
            d[1,i] = q_ii - delta
            d[3,i] = from_pocket
            d[4,i] = to_pocket
            d[5,i] = delta

    def plot_apply(self, d):
        """
        """
        fig, ax = pl.subplots(nrows=4, ncols=1, sharex=True, figsize=(10,10))
        # original signal & what is actually measured 
        ax[0].plot(d[0,:], 'k.', label='$n_i$')
        ax[0].plot(d[2,:], 'b+', label='$s_i$ (measured)')
        ax[0].legend(loc='best')
        ax[0].set_ylabel('signal')
        # pocket contents
        ax[1].plot(d[1,:], 'r.')
        ax[1].set_ylabel('pocket contents')
        # fill and flush functions 
        ax[2].plot(d[3,:], 'b.', label='flush')
        ax[2].plot(d[4,:], 'r+', label='fill')
        ax[2].legend(loc='best')
        # nett effect (flush - fill)
        ax[3].plot(d[3]-d[4,:], 'b.')
        ax[3].set_ylabel('net flux')

    def linearize(self, c, n):
        # if np.isscalar(c):
        #     c = np.array([c])
        # if np.isscalar(n):
        #     n = np.array([n])
        # c, n = np.array(c), np.array(n)
        d_from_pocket = self._d_flush(c)
        d_to_pocket = self._d_fill(c, n)
        d_delta = d_from_pocket - d_to_pocket
        d_delta_dc = d_delta[0]
        d_delta_dn = d_delta[1]
        ret = np.matrix([[1 + d_delta_dn, d_delta_dc], [-d_delta_dn, 1-d_delta_dc]])
        return ret

    def de_apply_global_lin(self, d):
        """
        """
        n_med = np.median(d[2,:])
        c_med = self.pocket_charge(n_med)
        print(f'n_med={n_med}  c_med={c_med}')

        M = self.linearize(c_med, n_med)
        M_inv = np.linalg.inv(M)

        dd = np.zeros_like(d)
        dd[2,:] = d[2,:] - n_med
        _,N = d.shape

        for i in range(N-1-30,1,-1):
            ds = dd[2,i]
            dc = dd[1,i]
            v = np.array(M_inv @ np.array([ds, dc])).squeeze()
            
            dd[0,i] = v[0]
            dd[1,i-1] = v[1]
            # dd[3,i] = 0.
            # dd[4,i] = to_pocket
            # dd[5,i] = delta
        dd[0] += n_med
        dd[1] += c_med#
        return dd

    def de_apply_no_lin(self, d):
        """
        """
        n_med = np.median(d[2,:])
        c_med = self.pocket_charge(n_med)
        print(f'n_med={n_med}  c_med={c_med}')

        # M = self.linearize(c_med, n_med)
        # M_inv = np.linalg.inv(M)

        dd = np.zeros_like(d)
        dd[2,:] = d[2,:]

        _,N = d.shape
        for i in range(N-1,1,-1):
            s = dd[2,i]
            c = dd[1,i]
            from_pocket = self._flush(c)
            to_pocket = self._fill(c,s)
            delta = from_pocket - to_pocket
            print(f'i={i} | c={c} s={s} | delta={delta}')
            dd[0,i] = s + delta
            dd[1,i-1] = c - delta
            dd[3,i] = from_pocket
            dd[4,i] = to_pocket
            dd[5,i] = delta
        return dd

    # def de_apply_local_lin(self, d):
    #     """
    #     """
    #     n_med = np.median(d[2])
    #     c_med = self.pocket_charge(n_med)
        
    #     dd = np.zeros_like(d)
    #     _,N = d.shape
    #     for i in range(N-1, 1, -1):
    #         signal = d[2,i]
    #         pocket_charge = d[1,i]
    #         delta = self._flush(pocket_charge) - self._fill(pocket_charge, signal)
    #         delta = np.array([delta, -delta])
    #         M = self.linearize(pocket_charge, signal)
    #         M_inv = np.linalg.inv(M)
    #         v = np.array([signal, pocket_charge]) - M_inv @ delta

    def de_apply(self, d):
        """
        """
       
        n_med = np.median(d[2,:])
        c_med = self.pocket_charge(n_med)
        print(f'n_med={n_med}  c_med={c_med}')
        _,N = d.shape
        # M = self.linearize(c_med, n_med)
        # M_inv = np.linalg.inv(M)

        dd = np.zeros_like(d)
        dd[2,:] = d[2,:]
        dd[1,N-1-30] = c_med
        for i in range(N-1-30,1,-1):
            s = dd[2,i]
            c = dd[1,i]
            delta = self._flush(c) - self._fill(c,s)
            print(s,c,delta)
            d_delta = self._d_flush(c) - self._d_fill(c,s)
            ddelta_dc = d_delta[0]
            ddelta_dn = d_delta[1]
            M = np.matrix([[1.+ddelta_dn, ddelta_dc], [-ddelta_dn, 1.-ddelta_dc]])
            M_inv = np.linalg.inv(M)
            v = (np.array(s,c) - np.array(M_inv @ np.array([delta,-delta]))).squeeze()
            dd[0,i] = v[0]
            dd[1,i-1] = v[1]
            dd[3,i] = dd[0,i] - dd[2,i]
        return dd


def two_point_function(flat):
    """
    """
    cov = []
    delta = np.arange(11).astype(int)
    mean_flat = flat.mean()
    ff = flat-mean_flat
    cov.append((ff**2).sum()/len(ff))
    for dd in delta[1:]:
        N = len(ff) - 2*dd
        cov.append((ff[dd:] * ff[:-dd]).sum() / N)
    return np.array(delta), np.array(cov)

def tpix_sensitivity_to_model(m, par_name='alpha', dpar=0.1, latex_par_name='\\alpha'):
    params = [(0, 'alpha', 0.1, '\\alpha'), 
              (1, 'cmax', 100., 'c_{max}'),
              (2, 'beta',  0.1, '\\beta'),
              (3, 'n0',   1000., 'n_0')]
    fig, axes = pl.subplots(nrows=2, ncols=2, figsize=(16,12), sharex=True)
    for i, par_name, dpar, latex_par_name in params:
        ii,jj = int(i/2), int(i%2)
        ax = axes[ii,jj]

        for skylev in [200, 1000]:
            logging.info(f'skylev: {skylev}')
            logging.info('simulated exposure')
            d = sim(flux=0, skylev=skylev, noise=True, size=9000000)
            logging.info('apply pocket model')
            m.apply(d)
            logging.info('two point function')
            delta, cov = two_point_function(d[2,:-30])

            d = sim(flux=0, skylev=skylev, noise=True, size=9000000)
            m.pars[par_name].full[:] += dpar
            logging.info('apply pocket model')
            m.apply(d)
            m.pars[par_name].full[:] -= dpar
            logging.info('two point function')
            ddelta, dcov = two_point_function(d[2,:-30])

            logging.info('ok. done.')
            ax.plot(delta, (dcov-cov)/(dpar), marker='.', ls=':', color=pl.cm.jet(int(i*256/6)), label=f'skylev={skylev}')
            # pl.plot(ddelta, dcov, marker='+', ls='', color=pl.cm.jet(int(i*256/6)))
            ax.legend(loc='best')
        ax.set_xlabel('pixel separation', fontsize=16)
        ax.set_ylabel(f'$\partial cov / \partial {latex_par_name}$', fontsize=16)

def plot_model_vs_skylev(m, skylev=[100, 500, 1000], colors=['k', 'c', 'r'], flux=1000, noise=False):

    pl.figure()
    for col, sklv in zip(colors, skylev):
        d = sim(noise=noise, skylev=sklv, flux=flux, size=1000)
        m.apply(d)
        pl.plot(d[5][450:550], ls=':', color=col, marker='.', label=f'skylev={sklv}')
    pl.legend()
    pl.xlabel('pixel')
    pl.ylabel('$\delta$')

class FillModel:

    def __init__(self, overscan_model):
        self.m = overscan_model
        data = np.unique(np.vstack(((np.array(self.m.dp.skylev), self.m.pars['pocket'].full[self.m.exp_index]))).T, axis=0)
        self.x = data[:,0]
        self.y = data[:,1]
        # self.pars = FitParameters([('a', 1), ('b', 1)])
        # self.pars['a'].full[:] = 1./6000.
        # self.pars['b'].full[:] = 0.3
        self.pars = FitParameters([('n0', 1), ('beta', 1)])
        self.pars['n0'].full[:] = self.m.pars['n0'].full[:]
        self.pars['beta'].full[:] = self.m.pars['beta'].full[:]

    def __call__(self, free_pars):
        self.pars.free = free_pars
        n0, beta = self.pars.full
        # n0, beta = self.pars.full
        pocket_max_charge, alpha = self.m.pars.full[0:2]
        # g_eq = np.power(beta * np.arctan(n0 * self.x) * 2. / np.pi, 1./alpha)
        g_eq = np.power(self.x/n0, beta)
        model_val = pocket_max_charge * g_eq / (1. + g_eq)
        # print(model_val)
        return self.y - model_val

def fit(ccdid=6, qid=1, plot=False, plot_exp_index=None):
    """
    """
    dp = load(ccdid=ccdid, qid=qid)
    m = OverscanModelNoDeltaMax(dp)
    # m.pars.fix('n0')
    # m.pars.fix('beta')
    fit_res = least_squares(m, m.pars.free, method='lm')

    fm = FillModel(m)
    fill_model_fit_res = least_squares(fm, fm.pars.free, method='lm') # , method='lm')
    print(fill_model_fit_res)
    m.pars['n0'].full[:] = fm.pars['n0'].full[:]
    m.pars['beta'].full[:] = fm.pars['beta'].full[:] * m.pars['alpha'].full[:]

    if plot:
        dd = m.overscan_data
        rr = m(m.pars.free).reshape(dd.shape)
        residuals = np.zeros_like(dd)
       
        # re-ordering rr so that it is ordered by mjd 
        i = np.argsort(m.dp.mjd_set)
        ddict = dict(zip(i,np.arange(len(i))))
        i = np.array([ddict[k] for k in m.dp.mjd_index])
        residuals[i,m.dp.j] = rr.flatten()[:]

        # i = np.argsort(m.dp.lastcol_set)
        # ddict = dict(zip(i,np.arange(len(i))))
        # i = np.array([ddict[k] for k in m.dp.lastcol_index])
        # residuals[i,m.dp.j] = rr.flatten()[:] # m.dp.temp

        # residuals[:] = rr[:]

        # fit residuals (global fit)
        pl.figure(figsize=(10,8))
        pl.imshow(residuals, aspect='auto')
        pl.colorbar()
        pl.title(f'overscan fit residuals (ccd={ccdid}, qid={qid})')
        pl.xlabel('col')
        pl.ylabel('residuals')

        if plot_exp_index is not None:
            for expidx in plot_exp_index:
                # fit residuals (exposure #12)
                fig, axes = pl.subplots(nrows=2, ncols=1, sharex=True)
                axes[0].plot(dd[expidx,:], 'k.', label='data')
                axes[0].plot(dd[expidx,] - rr[expidx,:], 'r-', label='model')
                axes[0].set_ylabel('overscan')
                axes[1].plot(rr[expidx,:], 'k.')
                axes[1].set_ylabel('residuals')
                axes[1].set_xlabel('col')
                axes[1].set_ylim((-3, 3))
                axes[0].set_title(f'overscan #{expidx} (ccd={ccdid}, qid={qid})')
                fig.subplots_adjust(hspace=0.05)

        # fill and flush functions
        m.plot_fill_flush()

        # fitted and predicted pocket charge
        pl.figure()
        pl.plot(dp.skylev, m.pars['pocket'].full[m.exp_index], 'k.')
        xx = np.linspace(0., dp.skylev.max(), 100)
        pl.plot(fm.x, fm.y-fm(fm.pars.free), 'b--')
        # m.pars['n0'].full[:] = fm.pars['n0'].full[:]
        # m.pars['beta'].full[:] = fm.pars['beta'].full[:]
        # # pl.plot(xx, m.pocket_charge(xx), 'r-')
        # # m.pars['beta'].full[:] = 0.5
        # pl.plot(xx, m.pocket_charge(xx), 'c-.')
        # m.pars['n0'].full[:] = 1000.
        # m.pars['beta'].full[:] = 0.4
        # pl.plot(xx, m.pocket_charge(xx), 'b--')
        # m.pars['n0'].full[:] = 1200.
        # m.pars['beta'].full[:] = 0.3
        # pl.plot(xx, m.pocket_charge(xx), 'g-')

        pl.title('pocket contents')
        pl.xlabel('sky level')
        pl.ylabel('pocket content (fitted)')

        fig, axes = pl.subplots(nrows=2, ncols=1, figsize=(8,8))
        cpocket = np.bincount(m.exp_index, weights=m.dp.overscan-m.dp.mean_trail_overscan)
        mjd_ = np.bincount(m.exp_index, weights=m.dp.mjd)
        cc_ = np.bincount(m.exp_index)
        axes[0].scatter(cpocket, m.pars['pocket'].full, c=mjd_/cc_)
        axes[0].set_ylabel('pocket charge (fitted)')
        axes[0].set_xlabel('pocket charge ($\sum_j ovscan_j$)')
        xx = np.linspace(cpocket.min()-10, cpocket.max()+10, 100)
        axes[0].plot(xx, xx, 'b--')
        axes[1].scatter(m.dp.mjd, (m.pars['pocket'].full/cpocket)[m.exp_index], c=m.dp.mjd)
        axes[1].set_xlabel('mjd')
        axes[1].set_ylabel('pocket charge (fitted) / pocket charge (overscan)')

    return m, fm

def fitall(plot=True):
    res = []
    for ccdid in range(1,17):
        for qid in range(1,5):
            print(f'ccd={ccdid} qid={qid}')
            m = fit(ccdid=ccdid, qid=qid, plot=False)
            res.append(m)

    if plot:
        alpha = np.array([r.pars['alpha'].full[0] for r in res])
        cmax  = np.array([r.pars['cmax'].full[0] for r in res])
        ccd = np.arange(1,17).repeat(4)
        qid = np.tile([1, 2, 3, 4], 16)
        fig, ax= pl.subplots(nrows=2, ncols=1, sharex=True)
        idx = alpha < 100
        ax[0].scatter(ccd[idx], alpha[idx], c=10*qid[idx], cmap=pl.cm.RdYlGn)
        ax[0].set_ylabel('$\\alpha$')
        ax[1].scatter(ccd[idx], cmax[idx], c=10*qid[idx], cmap=pl.cm.RdYlGn)
        ax[1].set_ylabel('$c_{max}$')
        ax[1].set_xlabel('CCD')
        fig.subplots_adjust(hspace=0.025)
    return res


class LogProb:

    def __init__(self, model):
        self.model = model
    
    def __call__(self, theta):
        res = self.model(theta)
        return (res**2).sum()

def mcmc():
    dp = load()
    model = OverscanModel(dp)
    logprob = LogProb(model)


def main(p, skylev=200, flux=1000, noise=False):
    """
    """
    d = sim(skylev=skylev, flux=flux, noise=noise)
    add_fq(d, fill , flush, p)
    plot_evol(d)
    plot_fill_flush(1000, p)
    return d


