Jenn Li Classifier
==================

1. Detection
	a. Peak detection (roughly higher than mean + pre-stimulus SD)
2. Features
	a. Duration
	b. Interval (time between peaks, largest between left or right)
3. Classifier
	a. Quadratic Discriminant Analysis (QDA)


Narendra Spectral Classifier
============================
1. Detection
	a. No event detection performed
2. Features
	b. Instantaneous dominant frequency (using BSA)
3. Classifier
	a. Frequency Bin (Gapes: 3.5-6 Hz, LTPs 6-9 Hz)


Multi-event Classifier
======================

**Note: Arguably, data should be pre-processed using top-hat filtering

1. Detection
	a. Top hat filtering (250 ms)
	b. Extract non-zero contiguous segments (**from unfiltered data**)
	c. Threshold event lengths between 50 and 500 ms
2. Features
	a. Duration
	b. Amplitude
	c. Left AND right interval
	d. PCA of warped segments (warping to equalize length)
	e. Max freq
3. Classifier
	a. XGBoost (default params)
