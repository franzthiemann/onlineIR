import math
import sys
import numpy as np
from pyspectra.readers.read_spc import read_spc_dir
import datetime
import os
import re
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.signal import find_peaks
import scipy as sp
import scipy.optimize

class IR_spectra:
    def __init__(self, folder):
        # Read data - Module has unnecessary print statements, blocking those
        old_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        df_spc, dict_spc = read_spc_dir(folder)
        sys.stdout = old_stdout
        print("Test")
        files = sorted(list(dict_spc.keys()))
        n_row = len(dict_spc[files[0]])
        n_col = len(files)
        self.times = np.zeros((n_col))
        self.freqs = np.array(dict_spc[files[0]].keys())
        self.data = np.zeros((n_col, n_row))
        for i, file in enumerate(files):
            path = os.path.join(folder, file)
            time = self._get_time(path)
            self.data[i, :] = dict_spc[file].values
            self.times[i] = time

    def _get_time(self, file):
        reg = r'Acquisition_Date_Time=(\d\d[.]\d\d[.]\d\d\d\d\s\d\d:\d\d:\d\d)'
        with open(file, "rb") as f:
            for line in f:
                decoded_line = line.decode("ascii", errors="ignore")
                match = re.search(reg, decoded_line)
                if match:
                    date_str = match.group(1)
                    date_obj = datetime.datetime.strptime(date_str, "%d.%m.%Y %H:%M:%S")
                    timestamp = date_obj.timestamp()
                    return timestamp
        return None

    def _get_ref(self, t_ref=0) -> int:
        # times from experiment start
        rel_times = self.times.copy() - self.times[0]
        for i, time in enumerate(rel_times):
            if time > t_ref:
                return i-1
        return len(rel_times)-1

    def _get_freq(self, ref_freq) -> int:
        # times from experiment start
        for i, freq in enumerate(self.freqs):
            if freq < ref_freq:
                return i
        return 0


    def plot3D(self, t_ref=0, freq_min=0, freq_max=math.inf):
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        print(fig)
        ref_ind = self._get_ref(t_ref)
        ref_spec = self.data[ref_ind, :]
        rel_times = self.times.copy()[ref_ind:] - self.times[ref_ind]
        rel_data = self.data[ref_ind:, :] - ref_spec
        # print("X:", rel_times.shape, "Y:", self.freqs.shape,  "Z:", rel_data.shape)
        X, Y = np.meshgrid(self.freqs, rel_times)
        ax.plot_surface(X, Y, rel_data, cmap=cm.coolwarm)
        # Calculate zoom
        f_max = min(freq_max, max(self.freqs))
        f_min = max(freq_min, min(self.freqs))
        ax.set_xlim(f_min, f_max)
        ax.set_xlabel("Wavenumber / cm$^{-1}$")
        ax.set_ylabel("Time / s")
        ax.set_zlabel("Intensity")
        plt.show()

    def plot_spectra(self, time, t_ref=0, freq_min=0, freq_max=math.inf, peaks=None):
        ind = self._get_ref(time)
        ref_ind = self._get_ref(t_ref)
        ref_spec = self.data[ref_ind, :]
        spec = self.data[ind, :] - ref_spec
        fig, ax = plt.subplots()
        ax.plot(self.freqs, spec)
        f_max = min(freq_max, max(self.freqs))
        f_min = max(freq_min, min(self.freqs))
        if peaks is not None:
            print("Searching for peaks")
            peaks_list, properties = find_peaks(spec, height=peaks)
            peak_x = self.freqs[peaks_list]
            for px in peak_x:
                plt.axvline(px, color="red", linestyle="--", alpha=0.7)
                if f_max > px > f_min:
                    print(f"Peak at {px}")
            print("Searching for negative peaks")
            nspec = [-x for x in spec]
            peaks_list, properties = find_peaks(nspec, height=peaks)
            peak_x = self.freqs[peaks_list]
            for px in peak_x:
                plt.axvline(px, color="green", linestyle="--", alpha=0.7)
                if f_max > px > f_min:
                    print(f"Peak at {px}")
        ax.set_xlim(f_min, f_max)
        ax.set_xlabel("Wavenumber / cm$^{-1}$")
        ax.set_ylabel("Intensity")
        plt.show()

    def plot_time(self, freq, width=1.0, t_ref=None, t_min=0, t_max=math.inf):
        freq_max_ind = self._get_freq(freq - width)
        freq_min_ind = self._get_freq(freq + width)
        rel_times = self.times - self.times[0]
        print(f"Accumulating between {self.freqs[freq_min_ind]} and {self.freqs[freq_max_ind]}")
        Y = np.zeros(len(self.times))
        for i, time in enumerate(rel_times):
            for j in range(freq_min_ind, freq_max_ind + 1):
                Y[i] += self.data[i, j]
        if t_ref is not None:
            ref_ind = self._get_ref(t_ref)
            Y = Y - Y[ref_ind]
        fig, ax = plt.subplots()
        ax.plot(rel_times, Y)
        t_max = min(t_max, max(rel_times))
        t_min = max(t_min, min(rel_times))
        ax.set_xlim(t_min, t_max)
        ax.set_xlabel("Time / s")
        ax.set_ylabel("Intensity")
        plt.show()

    def print_stats(self):
        print("IR Datast")
        print(f"{len(self.freqs)} frequencies between {self.freqs[0]} and {self.freqs[-1]}")
        start = datetime.datetime.fromtimestamp(self.times[0])
        end = datetime.datetime.fromtimestamp(self.times[-1])
        time = (self.times[-1]-self.times[0])/3600
        print(f"{len(self.times)} time steps between {start} and {end} covering {time} hours")

    def get_data(self, freq, t_ref=None, width=0, t_start=0, t_stop=math.inf, invert=False, normalize=0):
        freq_max_ind = self._get_freq(freq - width)
        freq_min_ind = self._get_freq(freq + width)
        rel_times = self.times - self.times[0]
        print(f"Accumulating between {self.freqs[freq_min_ind]} and {self.freqs[freq_max_ind]}")
        Y = np.zeros(len(self.times))
        for i, time in enumerate(rel_times):
            for j in range(freq_min_ind, freq_max_ind + 1):
                Y[i] += self.data[i, j]
        if t_ref is not None:
            ref_ind = self._get_ref(t_ref)
            Y = Y - Y[ref_ind]
        start_time = self._get_ref(t_start)
        end_time = self._get_ref(t_stop)
        X_crop = []
        Y_crop = []
        # print("times", start_time, end_time)
        for i in range(start_time, end_time):
            X_crop.append(self.times[i] - self.times[start_time])
            Y_crop.append(Y[i])
        if invert:
            if normalize == 0:
                normalize = 1
            Y_crop = [-1 * y for y in Y_crop]
        if normalize > 0:
            y_min = min(Y_crop)
            y_max = max(Y_crop)
            for i in range(len(Y_crop)):
                Y_crop[i] = (Y_crop[i] - y_min) * (1 / (y_max - y_min))
        if normalize == 2:
            def model_func(t, A1, K1, A2, K2, C, ):
                return A1 * np.exp(-K1 * t) + A2 * np.exp(-K2 * t) + C
            def fit_exp_nonlinear(x, y):
                opt_parms, parm_cov = sp.optimize.curve_fit(model_func, x, y, bounds=([0, 0, 0, 0, -1], [1, 0.1, 1, 0.1, 0]))
                # print(opt_parms, parm_cov)
                A1, K1, A2, K2, C = opt_parms
                return A1, K1, A2, K2, C
            A1, K1, A2, K2, C = fit_exp_nonlinear(X_crop, Y_crop)
            scale = -C
            print("Scale factor", scale)
            Y_crop = [scale + (1 - scale) * x for x in Y_crop]
        # print("lenth", len(X_crop), len(Y_crop))
        return X_crop, Y_crop