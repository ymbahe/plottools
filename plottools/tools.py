"""Collection of isolated tools."""

import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt
from pdb import set_trace
import scipy.stats.distributions as dist

class PlotDistribution:
    """Running mean/median of a given distribution."""

    def __init__(self, x, y, nbins=10, xrange=None, logbins=False, 
                 uncertainty=True, **kwargs):
        n_boot = 100

        if xrange is None:
            xmin, xmax = np.min(x), np.max(x)
        else:
            xmin, xmax = xrange

        if logbins:
            binedges = np.logspace(np.log10(xmin), np.log10(xmax),
                                   num=nbins+1, endpoint=True)
        else:
            binedges = np.linspace(xmin, xmax, num=nbins+1, endpoint=True)

        self.x, self.n, self.low, self.median, self.high = [
            np.copy(np.zeros(nbins)) for _i in range(5)]
        self.boot_low, self.boot_median, self.boot_high = [
            np.copy(np.zeros(nbins)) for _i in range(3)]

        rng = default_rng()

        # Go through each bin to find median and error
        for ibin in range(nbins):
            ind_bin = np.nonzero((x >= binedges[ibin]) &
                                 (x < binedges[ibin+1]))[0]
            self.n[ibin] = len(ind_bin)
            if self.n[ibin] == 0: continue

            bin_ys = y[ind_bin]
            self.low[ibin], self.median[ibin], self.high[ibin] = (
                np.percentile(y[ind_bin], [15.9, 50.0, 84.1]))
            self.x[ibin] = np.median(x[ind_bin])

            # Uncertainty on median via bootstrap resampling
            boot_medians = np.zeros(n_boot)
            for iboot in range(n_boot):
                boot_ys = rng.choice(bin_ys, size=len(ind_bin))
                boot_medians[iboot] = np.median(boot_ys)
             
            boot_percs = np.percentile(boot_medians, [15.9, 50.0, 84.1])
            self.boot_median[ibin] = boot_percs[1]
            self.boot_low[ibin], self.boot_high[ibin] = boot_percs[[0, 2]]

        # Plot the median line
        ind_plot = np.nonzero(self.n > 0)[0]
        plt.plot(self.x[ind_plot], self.median[ind_plot], **kwargs)

        if uncertainty:
            plt.fill_between(self.x[ind_plot],
                             self.boot_low[ind_plot], self.boot_high[ind_plot],
                             alpha=0.2, **kwargs)


class PlotFractionAboveThreshold:
    """Running fractions above a given threshold."""

    def __init__(self, x, y, threshold, nbins=10, npart=None, xrange=None,
                 logbins=False, min_width=None, 
                 plot=True, plot_actual=True, uncertainty=True,
                 mode='above', alpha_error=0.2, ci=0.68, min_count=5,
                 offset=0.0, **kwargs):
        n_boot = 100
        
        if xrange is None:
            xmin, xmax = np.min(x), np.max(x)
        else:
            xmin, xmax = xrange

        if npart is not None:
            ind_in_range = np.nonzero((x >= xmin) & (x < xmax))[0]
            ntot = len(ind_in_range)
            nbins = int(np.floor(ntot / npart))
            np_targ = int(ntot / nbins)
            if nbins < 5: nbins = 5
            separators = np.linspace(0, ntot, nbins+1, dtype=int)
            sorter = np.argsort(x[ind_in_range])

            if min_width is not None:
                separators = [0]
                bin_start = xrange[0]
                nbins = 0
                curr_offset = 0
                while(True):
                    trial_end = min(curr_offset + np_targ, ntot-1)
                    x_part = x[ind_in_range[sorter[trial_end]]]
                    if x_part - bin_start > min_width:
                        curr_offset += trial_end
                        separators.append(curr_offset)
                        bin_start = x_part
                    else:
                        curr_offset = np.searchsorted(
                            x[ind_in_range[sorter]], bin_start + min_width)
                        separators.append(curr_offset)
                        bin_start = x[ind_in_range[sorter[curr_offset]]]
                    if separators[-1] >= ntot:
                        break
                nbins = len(separators) - 1
                separators = np.clip(separators, 0, ntot-1)
        else:
            if logbins:
                binedges = np.logspace(np.log10(xmin), np.log10(xmax),
                                       num=nbins+1, endpoint=True)
            else:
                binedges = np.linspace(xmin, xmax, num=nbins+1, endpoint=True)
        
        self.n = np.zeros(nbins)
        self.n_succ = np.zeros(nbins) + np.nan
        self.f_succ = np.zeros(nbins) + np.nan
        self.x = np.zeros(nbins) + np.nan

        # Need to analyse each bin in turn
        for ibin in range(nbins):
            
            if npart is not None:
                ind_bin = ind_in_range[sorter[np.arange(separators[ibin],
                                                        separators[ibin+1])]]
            else:
                ind_bin = np.nonzero((x >= binedges[ibin]) &
                                     (x < binedges[ibin+1]))[0]
            self.n[ibin] = len(ind_bin)
            if self.n[ibin] == 0: continue

            self.x[ibin] = np.median(x[ind_bin])

            if mode == 'above':
                ind_succ = (np.nonzero(y[ind_bin] > threshold))[0]   
            elif mode == 'at':
                ind_succ = (np.nonzero(y[ind_bin] == threshold))[0]
            elif mode == 'below':
                ind_succ = (np.nonzero(y[ind_bin] < threshold))[0]
            else:
                print(f"Illegal mode '{mode}'.")
                set_trace()

            self.n_succ[ibin] = len(ind_succ)            
            self.f_succ[ibin] = self.n_succ[ibin] / self.n[ibin]
        
        # Calculate binomial uncertainties, if desired
        if uncertainty:
            self.p_lower = dist.beta.ppf(
                (1 - ci) / 2., self.n_succ + 1, self.n - self.n_succ + 1)
            self.p_upper = dist.beta.ppf(
                1 - (1 - ci) /2. , self.n_succ + 1, self.n - self.n_succ + 1)
        else:
            self.p_lower = None
            self.p_upper = None

        # Plot the result, if desired
        if plot:        
            ind_plot = np.nonzero(self.n > min_count)[0]    
           
            if plot_actual:
                plt.plot(self.x[ind_plot],
                         self.f_succ[ind_plot] + offset,
                         marker='o', markersize=1, **kwargs)
                
            if uncertainty:            
                plt.fill_between(
                    self.x[ind_plot], self.p_lower[ind_plot] + offset,
                    self.p_upper[ind_plot] + offset, alpha=alpha_error,
                    **kwargs)
                
