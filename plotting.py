#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Belle-II module
import copy
import math

import pandas
import numpy
import matplotlib.pyplot as plt
import matplotlib.artist
import matplotlib.figure
import matplotlib.gridspec
import matplotlib.colors
import matplotlib.patches
import matplotlib.ticker
import matplotlib

# framework includes
import histogram as histo

class Plotter(object):
    """
    Base class for all Plotters.
    """

    # stupid workaround for doxygen refusing to document things

    #: @fn set_errorbar_options(errorbar_kwargs)
    #: Overrides default errorbar options for datapoint errorbars

    #: @var xscale
    #: limit scale
    #: @var yscale
    #: limit scale

    #: Plots added to the axis so far
    plots = None
    #: Labels of the plots added so far
    labels = None
    #: Minimum x value
    xmin = None
    #: Maximum x value
    xmax = None
    #: Minimum y value
    ymin = None
    #: Maximum y value
    ymax = None
    yscale = 0.0
    xscale = 0.0
    #: figure which is used to draw
    figure = None
    #: Main axis which is used to draw
    axis = None

    def __init__(self, figure=None, axis=None):
        """
        Creates a new figure and axis if None is given, sets the default plot parameters
        @param figure default draw figure which is used
        @param axis default draw axis which is used
        """
        print("Create new figure for class " + str(type(self)))
        if figure is None:
            self.figure = matplotlib.figure.Figure(figsize=(32, 18))
            self.figure.set_tight_layout(False)
        else:
            self.figure = figure

        if axis is None:
            self.axis = self.figure.add_subplot(1, 1, 1)
        else:
            self.axis = axis

        self.plots = []
        self.labels = []
        self.xmin, self.xmax = float(0), float(1)
        self.ymin, self.ymax = float(0), float(1)
        #: y limit scale
        self.yscale = 0.1
        #: x limit scale
        self.xscale = 0.0

        #: Default keyword arguments for plot function
        self.plot_kwargs = None
        #: Default keyword arguments for errorbar function
        self.errorbar_kwargs = None
        #: Default keyword arguments for errorband function
        self.errorband_kwargs = None
        #: Default keyword arguments for fill_between function
        self.fill_kwargs = None

        self.set_plot_options()
        self.set_errorbar_options()
        self.set_errorband_options()
        self.set_fill_options()

    def add_subplot(self, gridspecs):
        """
        Adds a new subplot to the figure, updates all other axes
        according to the given gridspec
        @param gridspecs gridspecs for all axes including the new one
        """
        for gs, ax in zip(gridspecs[:-1], self.figure.axes):
            ax.set_position(gs.get_position(self.figure))
            ax.set_subplotspec(gs)
        axis = self.figure.add_subplot(gridspecs[-1], sharex=self.axis)
        return axis

    def save(self, filename):
        """
        Save the figure into a file
        @param filename of the file
        """
        print("Save figure for class " + str(type(self)))
        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
        canvas = FigureCanvas(self.figure)
        canvas.print_figure(filename, dpi=50)
        return self

    def set_plot_options(self, plot_kwargs={'linestyle': ''}):
        """
        Overrides default plot options for datapoint plot
        @param plot_kwargs keyword arguments for the plot function
        """
        self.plot_kwargs = copy.copy(plot_kwargs)
        return self

    def set_errorbar_options(self, errorbar_kwargs={'fmt': '.', 'elinewidth': 3, 'alpha': 1}):
        """
        Overrides default errorbar options for datapoint errorbars
        @param errorbar_kwargs keyword arguments for the errorbar function
        """
        self.errorbar_kwargs = copy.copy(errorbar_kwargs)
        return self

    def set_errorband_options(self, errorband_kwargs={'alpha': 0.5}):
        """
        Overrides default errorband options for datapoint errorband
        @param errorbar_kwargs keyword arguments for the fill_between function
        """
        self.errorband_kwargs = copy.copy(errorband_kwargs)
        return self

    def set_fill_options(self, fill_kwargs=None):
        """
        Overrides default fill_between options for datapoint errorband
        @param fill_kwargs keyword arguments for the fill_between function
        """
        self.fill_kwargs = copy.copy(fill_kwargs)
        return self

    def _plot_datapoints(self, axis, x, y, xerr=None, yerr=None):
        """
        Plot the given datapoints, with plot, errorbar and make a errorband with fill_between
        @param x coordinates of the data points
        @param y coordinates of the data points
        @param xerr symmetric error on x data points
        @param yerr symmetric error on y data points
        """
        p = e = f = None
        plot_kwargs = copy.copy(self.plot_kwargs)
        errorbar_kwargs = copy.copy(self.errorbar_kwargs)
        errorband_kwargs = copy.copy(self.errorband_kwargs)
        fill_kwargs = copy.copy(self.fill_kwargs)

        if plot_kwargs is None or 'color' not in plot_kwargs:
            color = next(axis._get_lines.prop_cycler)
            color = color['color']
            plot_kwargs['color'] = color
        else:
            color = plot_kwargs['color']
        color = matplotlib.colors.ColorConverter().to_rgb(color)
        patch = matplotlib.patches.Patch(color=color, alpha=0.5)
        patch.get_color = patch.get_facecolor
        patches = [patch]

        if plot_kwargs is not None:
            p, = axis.plot(x, y, rasterized=True, **plot_kwargs)
            patches.append(p)

        if errorbar_kwargs is not None and (xerr is not None or yerr is not None):
            if 'color' not in errorbar_kwargs:
                errorbar_kwargs['color'] = color
            if 'ecolor' not in errorbar_kwargs:
                errorbar_kwargs['ecolor'] = [0.5 * x for x in color]
            e = axis.errorbar(x, y, xerr=xerr, yerr=yerr, rasterized=True, **errorbar_kwargs)
            patches.append(e)

        if errorband_kwargs is not None and yerr is not None:
            if 'color' not in errorband_kwargs:
                errorband_kwargs['color'] = color
            if xerr is not None:
                # Ensure that xerr and yerr are iterable numpy arrays
                xerr = x + xerr - x
                yerr = y + yerr - y
                for _x, _y, _xe, _ye in zip(x, y, xerr, yerr):
                    axis.add_patch(matplotlib.patches.Rectangle((_x - _xe, _y - _ye), 2 * _xe, 2 * _ye, rasterized=True,
                                                                **errorband_kwargs))
            else:
                f = axis.fill_between(x, y - yerr, y + yerr, interpolate=True, rasterized=True, **errorband_kwargs)

        if fill_kwargs is not None:
            axis.fill_between(x, y, 0, rasterized=True, **fill_kwargs)

        return (tuple(patches), p, e, f)

    def add(self, *args, **kwargs):
        """
        Add a new plot to this plotter
        """
        return NotImplemented

    def finish(self, *args, **kwargs):
        """
        Finish plotting and set labels, legends and stuff
        """
        return NotImplemented

    def scale_limits(self):
        """
        Scale limits to increase distance to boundaries
        """
        self.ymin *= 1.0 - math.copysign(self.yscale, self.ymin)
        self.ymax *= 1.0 + math.copysign(self.yscale, self.ymax)
        self.xmin *= 1.0 - math.copysign(self.xscale, self.xmin)
        self.xmax *= 1.0 + math.copysign(self.xscale, self.xmax)
        return self


class Distribution(Plotter):
    """
    Plots distribution of a quantity
    """

    def __init__(self, figure=None, axis=None, normed_to_all_entries=False, normed_to_bin_width=False,
                 keep_first_binning=False, range_in_std=None):
        """
        Creates a new figure and axis if None is given, sets the default plot parameters
        @param figure default draw figure which is used
        @param axis default draw axis which is used
        @param normed true if histograms should be normed before drawing
        @param keep_first_binning use the binning of the first distribution for further plots
        @param range_in_std show only the data in a windows around +- range_in_std * standard_deviation around the mean
        """
        super(Distribution, self).__init__(figure, axis)
        #: Normalize histograms before drawing them
        self.normed_to_all_entries = normed_to_all_entries
        #: Normalize histograms before drawing them
        self.normed_to_bin_width = normed_to_bin_width
        #: Show only a certain range in terms of standard deviations of the data
        self.range_in_std = range_in_std
        # if self.normed_to_all_entries or self.normed_to_bin_width:
        #: size in x/y
        self.ymin = float(0)
        #: size in x/y
        self.ymax = float('-inf')
        #: size in x/y
        self.xmin = float('inf')
        #: size in x/y
        self.xmax = float('-inf')
        #: Keep first binning if user wants so
        self.keep_first_binning = keep_first_binning
        #: first binning
        self.first_binning = None
        #: x axis label
        self.x_axis_label = ''

    def add(self, data, column, mask=None, weight_column=None, label=None):
        """
        Add a new distribution to the plots
        @param data pandas.DataFrame containing all data
        @param column which is used to calculate distribution histogram
        @param mask boolean numpy.array defining which events are used for the histogram
        @param weight_column column in data containing the weights for each event
        """
        if mask is None:
            mask = numpy.ones(len(data)).astype('bool')

        bins = 100
        if self.keep_first_binning and self.first_binning is not None:
            bins = self.first_binning
        hists = histo.Histograms(data, column, {'Total': mask}, weight_column=weight_column,
                                     bins=bins, equal_frequency=False, range_in_std=self.range_in_std)
        if self.keep_first_binning and self.first_binning is None:
            self.first_binning = hists.bins
        hist, hist_error = hists.get_hist('Total')

        if self.normed_to_all_entries:
            normalization = float(numpy.sum(hist))
            hist = hist / normalization
            hist_error = hist_error / normalization

        if self.normed_to_bin_width:
            hist = hist / hists.bin_widths
            hist_error = hist_error / hists.bin_widths

        self.xmin, self.xmax = min(hists.bin_centers.min(), self.xmin), max(hists.bin_centers.max(), self.xmax)
        self.ymin = numpy.nanmin([hist.min(), self.ymin])
        self.ymax = numpy.nanmax([(hist + hist_error).max(), self.ymax])

        p = self._plot_datapoints(self.axis, hists.bin_centers, hist, xerr=hists.bin_widths / 2, yerr=hist_error)
        self.plots.append(p)
        self.x_axis_label = column

        appendix = ''
        if self.ymax <= self.ymin or self.xmax <= self.xmin:
            appendix = ' No data to plot!'

        if label is None:
            self.labels.append(column + appendix)
        else:
            self.labels.append(label + appendix)
        return self

    def finish(self):
        """
        Sets limits, title, axis-labels and legend of the plot
        """
        self.axis.set_title("Distribution Plot")
        self.axis.get_xaxis().set_label_text(self.x_axis_label)

        self.axis.legend([x[0] for x in self.plots], self.labels, loc='best', fancybox=True, framealpha=0.5)

        if self.ymax <= self.ymin or self.xmax <= self.xmin:
            self.axis.set_xlim((0., 1.))
            self.axis.set_ylim((0., 1.))
            self.axis.text(0.36, 0.5, 'No data to plot', fontsize=60, color='black')
            return self

        self.scale_limits()

        self.axis.set_xlim((self.xmin, self.xmax))
        self.axis.set_ylim((self.ymin, self.ymax))

        if self.normed_to_all_entries and self.normed_to_bin_width:
            self.axis.get_yaxis().set_label_text('# Entries per Bin / (# Entries * Bin Width)')
        elif self.normed_to_all_entries:
            self.axis.get_yaxis().set_label_text('# Entries per Bin / # Entries')
        elif self.normed_to_bin_width:
            self.axis.get_yaxis().set_label_text('# Entries per Bin / Bin Width')
        else:
            self.axis.get_yaxis().set_label_text('# Entries per Bin')

        return self



class Difference(Plotter):
    """
    Plots the difference between two histograms
    """
    #: @var xmax
    #: Maximum x value
    #: @var ymax
    #: Maximum y value
    #: @var ymin
    #: min y value
    #: @var x_axis_label
    #: Label on x axis
    #: @var normed
    #: Minuend and subtrahend are normed before comparing them if this is true
    #: @var shift_to_zero
    #: Mean difference is shifted to zero (removes constant offset) if this is true

    def __init__(self, figure=None, axis=None, normed=False, shift_to_zero=False):
        """
        Creates a new figure and axis if None is given, sets the default plot parameters
        @param figure default draw figure which is used
        @param axis default draw axis which is used
        @param normed normalize minuend and subtrahend before comparing them
        @param shift_to_zero mean difference is shifted to zero, to remove constant offset due to e.g. different sample sizes
        """
        super(Difference, self).__init__(figure, axis)
        self.normed = normed
        self.shift_to_zero = shift_to_zero
        if self.normed:
            self.ymin = -0.01
            self.ymax = 0.01
        else:
            self.ymin = -1
            self.ymax = 1

    def add(self, data, column, minuend_mask, subtrahend_mask, weight_column=None, label=None):
        """
        Add a new difference plot
        @param data pandas.DataFrame containing all data
        @param column which is used to calculate distribution histogram
        @param minuend_mask boolean numpy.array defining which events are for the minuend histogram
        @param subtrahend_mask boolean numpy.array defining which events are for the subtrahend histogram
        @param weight_column column in data containing the weights for each event
        @param label label for the legend if None, the column name is used
        """
        hists = histo.Histograms(data, column, {'Minuend': minuend_mask, 'Subtrahend': subtrahend_mask},
                                     weight_column=weight_column, equal_frequency=False)
        minuend, minuend_error = hists.get_hist('Minuend')
        subtrahend, subtrahend_error = hists.get_hist('Subtrahend')

        difference_error = histo.poisson_error(minuend + subtrahend)
        if self.normed:
            difference_error = difference_error / (numpy.sum(minuend) + numpy.sum(subtrahend))
            minuend = minuend / numpy.sum(minuend)
            subtrahend = subtrahend / numpy.sum(subtrahend)
        difference = minuend - subtrahend

        if self.shift_to_zero:
            difference = difference - numpy.mean(difference)

        self.xmin, self.xmax = min(hists.bin_centers.min(), self.xmin), max(hists.bin_centers.max(), self.xmax)
        self.ymin = min((difference - difference_error).min(), self.ymin)
        self.ymax = max((difference + difference_error).max(), self.ymax)

        p = self._plot_datapoints(self.axis, hists.bin_centers, difference, xerr=hists.bin_widths / 2, yerr=difference_error)
        self.plots.append(p)
        if label is None:
            self.labels.append(label)
        else:
            self.labels.append(column)
        self.x_axis_label = column
        return self

    def finish(self, line_color='black'):
        """
        Sets limits, title, axis-labels and legend of the plot
        """
        self.axis.plot((self.xmin, self.xmax), (0, 0), color=line_color, linewidth=4, rasterized=True)
        self.scale_limits()
        self.axis.set_xlim((self.xmin, self.xmax))
        self.axis.set_ylim((self.ymin, self.ymax))
        self.axis.set_title("Difference Plot")
        self.axis.get_yaxis().set_major_locator(matplotlib.ticker.MaxNLocator(5))
        self.axis.get_xaxis().set_label_text(self.x_axis_label)
        self.axis.get_yaxis().set_label_text('Difference')
        self.axis.legend([x[0] for x in self.plots], self.labels, loc='best', fancybox=True, framealpha=0.5)
        return self



class Overtraining(Plotter):
    """
    Create TMVA-like overtraining control plot for a classification training
    """

    #: figure which is used to draw
    figure = None
    #: Main axis which is used to draw
    axis = None
    #: Axis which shows the difference between training and test signal
    axis_d1 = None
    #: Axis which shows the difference between training and test background
    axis_d2 = None

    def __init__(self, figure=None):
        """
        Creates a new figure if None is given, sets the default plot parameters
        @param figure default draw figure which is used
        """
        if figure is None:
            self.figure = matplotlib.figure.Figure(figsize=(32, 18))
            self.figure.set_tight_layout(True)
        else:
            self.figure = figure

        gs = matplotlib.gridspec.GridSpec(5, 1)
        self.axis = self.figure.add_subplot(gs[:3, :])
        self.axis_d1 = self.figure.add_subplot(gs[3, :], sharex=self.axis)
        self.axis_d2 = self.figure.add_subplot(gs[4, :], sharex=self.axis)

        super(Overtraining, self).__init__(self.figure, self.axis)

    def add(self, data, column, train_mask, test_mask, signal_mask, bckgrd_mask, weight_column=None):
        """
        Add a new overtraining plot, I recommend to raw only one overtraining plot at the time,
        otherwise there are too many curves in the plot to recognize anything in the plot.
        @param data pandas.DataFrame containing all data
        @param column which is used to calculate distribution histogram
        @param train_mask boolean numpy.array defining which events are training events
        @param test_mask boolean numpy.array defining which events are test events
        @param signal_mask boolean numpy.array defining which events are signal events
        @param bckgrd_mask boolean numpy.array defining which events are background events
        @param weight_column column in data containing the weights for each event
        """
        distribution = Distribution(self.figure, self.axis, normed_to_all_entries=True)

        distribution.set_plot_options(self.plot_kwargs)
        distribution.set_errorbar_options(self.errorbar_kwargs)
        distribution.set_errorband_options(self.errorband_kwargs)
        distribution.add(data, column, test_mask & signal_mask, weight_column)
        distribution.add(data, column, test_mask & bckgrd_mask, weight_column)

        distribution.set_plot_options(
            {'color': distribution.plots[0][0][0].get_color(), 'linestyle': '-', 'lw': 4, 'drawstyle': 'steps-mid'})
        distribution.set_fill_options({'color': distribution.plots[0][0][0].get_color(), 'alpha': 0.5, 'step': 'mid'})
        distribution.set_errorbar_options(None)
        distribution.set_errorband_options(None)
        distribution.add(data, column, train_mask & signal_mask, weight_column)
        distribution.set_plot_options(
            {'color': distribution.plots[1][0][0].get_color(), 'linestyle': '-', 'lw': 4, 'drawstyle': 'steps-mid'})
        distribution.set_fill_options({'color': distribution.plots[1][0][0].get_color(), 'alpha': 0.5, 'step': 'mid'})
        distribution.add(data, column, train_mask & bckgrd_mask, weight_column)

        distribution.labels = ['Test-Signal', 'Test-Background', 'Train-Signal', 'Train-Background']
        distribution.finish()

        self.plot_kwargs['color'] = distribution.plots[0][0][0].get_color()
        difference_signal = Difference(self.figure, self.axis_d1, shift_to_zero=True, normed=True)
        difference_signal.set_plot_options(self.plot_kwargs)
        difference_signal.set_errorbar_options(self.errorbar_kwargs)
        difference_signal.set_errorband_options(self.errorband_kwargs)
        difference_signal.add(data, column, train_mask & signal_mask, test_mask & signal_mask, weight_column)
        self.axis_d1.set_xlim((difference_signal.xmin, difference_signal.xmax))
        self.axis_d1.set_ylim((difference_signal.ymin, difference_signal.ymax))
        difference_signal.plots = difference_signal.labels = []
        difference_signal.finish(line_color=distribution.plots[0][0][0].get_color())

        self.plot_kwargs['color'] = distribution.plots[1][0][0].get_color()
        difference_bckgrd = Difference(self.figure, self.axis_d2, shift_to_zero=True, normed=True)
        difference_bckgrd.set_plot_options(self.plot_kwargs)
        difference_bckgrd.set_errorbar_options(self.errorbar_kwargs)
        difference_bckgrd.set_errorband_options(self.errorband_kwargs)
        difference_bckgrd.add(data, column, train_mask & bckgrd_mask, test_mask & bckgrd_mask, weight_column)
        self.axis_d2.set_xlim((difference_bckgrd.xmin, difference_bckgrd.xmax))
        self.axis_d2.set_ylim((difference_bckgrd.ymin, difference_bckgrd.ymax))
        difference_bckgrd.plots = difference_bckgrd.labels = []
        difference_bckgrd.finish(line_color=distribution.plots[1][0][0].get_color())

        try:
            import scipy.stats
            # Kolmogorov smirnov test
            if len(data[column][train_mask & signal_mask]) == 0 or len(data[column][test_mask & signal_mask]) == 0:
                print("Cannot calculate kolmogorov smirnov test for signal due to missing data")
            else:
                ks = scipy.stats.ks_2samp(data[column][train_mask & signal_mask], data[column][test_mask & signal_mask])
                props = dict(boxstyle='round', edgecolor='gray', facecolor='white', linewidth=0.1, alpha=0.5)
                self.axis_d1.text(0.1, 0.9, r'signal (train - test) difference $p={:.2f}$'.format(ks[1]), fontsize=36, bbox=props,
                                  verticalalignment='top', horizontalalignment='left', transform=self.axis_d1.transAxes)
            if len(data[column][train_mask & bckgrd_mask]) == 0 or len(data[column][test_mask & bckgrd_mask]) == 0:
                print("Cannot calculate kolmogorov smirnov test for background due to missing data")
            else:
                ks = scipy.stats.ks_2samp(data[column][train_mask & bckgrd_mask], data[column][test_mask & bckgrd_mask])
                props = dict(boxstyle='round', edgecolor='gray', facecolor='white', linewidth=0.1, alpha=0.5)
                self.axis_d2.text(0.1, 0.9, r'background (train - test) difference $p={:.2f}$'.format(ks[1]), fontsize=36,
                                  bbox=props,
                                  verticalalignment='top', horizontalalignment='left', transform=self.axis_d2.transAxes)
        except ImportError:
            print("Cannot calculate kolmogorov smirnov test please install scipy!")

        return self

    def finish(self):
        """
        Sets limits, title, axis-labels and legend of the plot
        """
        self.axis.set_title("Overtraining Plot")
        self.axis_d1.set_title("")
        self.axis_d2.set_title("")
        matplotlib.artist.setp(self.axis.get_xticklabels(), visible=False)
        matplotlib.artist.setp(self.axis_d1.get_xticklabels(), visible=False)
        self.axis.get_xaxis().set_label_text('')
        self.axis_d1.get_xaxis().set_label_text('')
        self.axis_d2.get_xaxis().set_label_text('Classifier Output')
        return self
