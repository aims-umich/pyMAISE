class Device(object):
    """
    Device object for keeping track of memory and number of jobs that pyMAISE launches
    on it.

    Parameters
    ----------
    id: int or tensorflow.config.PhysicalDevice
        ID for device.
    free_memory: int
        Current memory on the device in MB.
    is_gpu: bool
        GPU flag.
    """

    def __init__(self, id, free_memory, is_gpu):
        # Save data
        self._id = id
        self._free_memory = free_memory
        self._starting_memory = self._free_memory
        self._is_gpu = is_gpu
        self._num_jobs = 0

    # =======================================================================
    # Overloads

    def __repr__(self):
        return "Device(id={}, free_memory={} MB, is_gpu={})".format(
            self._id, self._free_memory, self._is_gpu
        )

    # =======================================================================
    # Setters/Getters

    @property
    def id(self):
        return self._id

    @property
    def free_memory(self):
        return self._free_memory

    @property
    def is_gpu(self):
        return self._is_gpu

    @property
    def starting_memory(self):
        return self._starting_memory

    @property
    def num_jobs(self):
        return self._num_jobs

    @free_memory.setter
    def free_memory(self, free_memory):
        self._free_memory = free_memory

    @num_jobs.setter
    def num_jobs(self, num_jobs):
        self._num_jobs = num_jobs
