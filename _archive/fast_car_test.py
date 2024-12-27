"""
Chop each data file into chunks that can be processed in parallel
E.g. Chop 32 files into 10 chunks each
    Multiprocess parallelization will be performed over chunks
    such that for each chunk:
    1) Data will be loaded serially from each channel
    2) Once data is loaded, averaging will be performed and
    3) Referencing using the average performed on data from each electrode
    4) The referenced data will be written to the respective slice of the
        HDF5 array
        a) Access disputes will be settled using a global variable

    This will prevent multiple I/O (writing raw data to HDF5 and writing
    it again after referencing), but will require upfront splitting of
    data files into chunks
"""

# Find
