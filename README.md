# onlineIR
Allows .spc files exported from Mettler Toledo iC IR to be read, plotted and evaluated in Python and Jupyter notebooks

## Usage
### Reading the spectra
Read the folder by initializing the IR_spectra object\
```spec = IR_spectra(<folder>)```

### Checking the Spectra
`IR_spectra.print_stats()`\
Returns the acquisition time and duration along with the time samples taken during this time.
Further the spectral range and resolution is displayed.

### 3D Plot of the spectra
`IR_spectra.print_stats()`\
Creates a 3D Plot of the gathered data (Intensity as a function of time and wave number).
When using in a jupyter notebook, use `%matplotlib widget` to get an interactive view.
The following optional arguments can be supplied:
- `t_ref = 0` set the time in seconds to be used as a reference for the intensities
- `freq_min = 0` set the minimum frequency to be included in the plot
- `freq_max = inf` set the maximum frequency to be included in the plot

### Spectra at certain point in time
`IR_spectra.plot_spectra(<time>)`\
Plot the spectra at a certain time `<time>` (given in seconds).
Optional arguments:
- `t_ref = 0` set the time in seconds to be used as a reference for the intensities
- `freq_min = 0` set the minimum frequency to be included in the plot
- `freq_max = inf` set the maximum frequency to be included in the plot
- `peaks = None` if given a numeric value: Highlights peaks above this value and prints a list of peaks

### Time dependence of a peak
`IR_spectra.plot_time(<freq>)`\
Plots the time dependence of a given signal at `<freq>`.
Optional arguments:
- `width = 1` include wave numbers above and below the given signal (helps when peaks shift due to temperature)
- `t_ref = 0` set the time in seconds to be used as a reference for the intensities
- `t_min = 0` set start time in seconds
- `t_max = inf` set end time in seconds

### Extract data for further processing
`IR_spectra.get_data(<freq>)`
Returns the data for the given signal as a x- and y-list.
Optional arguments:
- `t_ref = 0` set the time in seconds to be used as a reference for the intensities
- `width = 0` include wave numbers above and below the given signal (helps when peaks shift due to temperature)
- `t_start = 0` set start time in seconds
- `t_stop = inf` set end time in seconds
- `invert = False` invert the signal (convert from absorbance to transmission), automatically enforces normalization
- `normalize` 
  - `normalize = 0` no normalization (default)
  - `normalize = 1` simple normalization to range of 0 to 1 by min and max value
  - `normalize = 2` normalization via extrapolation using predefined extrapolation function $I(t) = A_1 e^{-k_1 t} + A_2 e^{-k_2 t} + C$