from multiprocessing import Process, Manager
import subprocess as sp

import numpy as np
from tensorflow.config import list_physical_devices

import pyMAISE.settings as settings
from .device import Device


class ProcessPool(object):
    """
    Process pool object for multiprocessing.

    Finds all eligible GPUs as Device objects.
    """

    def __init__(self):
        self._num_processes = 0
        self._processes = {}

        with Manager() as manager:
            self._devices = manager.list()

            # Setup all devices
            process = Process(target=self._setup_devices, args=(self._devices,))
            process.start()
            process.join()
            assert process.exitcode == 0
            process.terminate()

            self._devices = list(self._devices)

    # =======================================================================
    # Methods

    def check_model_size(self, job_memory):
        """
        Check if model fits within one GPU.

        Parameters
        ----------
        job_memory: int
            Size of the model in MB.
        """
        # Iterate through devices and make sure we can
        # actually train the model
        for device in self._devices:
            if device.starting_memory > job_memory:
                return True

        raise RuntimeError(
            f"Model size ({job_memory} MB) doesn't fit in any available GPUs"
        )

    def find_eligible_device(self, job_memory):
        """
        Find an eligible device for a given job. Attempts to find a device that has
        enough memory for the job. If more then one device is available then it will
        choose the one with the least amount of jobs.

        Parameters
        ----------
        job_memory: int
            Size of the model in MB.

        Returns
        -------
        idx: int
            Index for the device within ``ProcessPool`` device list.
        """
        # Iterate through devices to find one with enough memory
        idx = None
        num_jobs = np.inf
        for i, device in enumerate(self._devices):
            if (
                device.free_memory - device.num_jobs * 400 > job_memory + 400
                and device.num_jobs < num_jobs
                and device.num_jobs < settings.values.max_models_per_device
            ):
                idx = i
                num_jobs = device.num_jobs

        return idx

    def submit_process(self, target, args, device_idx, job_memory):
        """
        Submit a process to a device.

        Parameters
        ----------
        target: callable
            Target function for execution on the device.
        args: tuple
            Tuple of parameters for the target function.
        device_idx: int
            Index into list of devices within the ``ProcessPool`` object.
        job_memory: int
            Size of the model in MB.

        Returns
        -------
        pid: int
            Integer ID of the process that was started.
        """
        # Create and start process
        process = Process(target=target, args=args)
        process.start()

        # Add process to pool and reserve GPU memory
        self._processes[process.pid] = (process, device_idx, job_memory)
        self._devices[device_idx].free_memory -= job_memory
        self._devices[device_idx].num_jobs += 1

        self._num_processes += 1

        return process.pid

    def is_alive(self, pid):
        """
        Check if a process is alive.

        Parameters
        ----------
        pid: int
            Integer ID of process.

        Returns
        -------
        is_alive: bool
            Whether the process is running or not.
        """
        # Get process from pool
        process, device_idx, job_memory = self._processes[pid]

        # Check join
        process.join(timeout=0.1)

        # Check if process is alive
        if not process.is_alive():
            assert process.exitcode == 0
            process.terminate()

            # Remove process from pool
            self._processes.pop(pid)

            # Release GPU memory
            self._devices[device_idx].free_memory += job_memory

            self._devices[device_idx].num_jobs -= 1
            self._num_processes -= 1

            return False
        return True

    # =======================================================================
    # Static Methods
    @staticmethod
    def _setup_devices(devices):
        """
        Setup GPU device list.

        Parameters
        ----------
        devices: list
            Empty list. After this function runs this will be a list of
            Device objects that are GPUs.
        """
        # Get available GPUs
        gpu_ids = list_physical_devices("GPU")

        if gpu_ids:
            # Run with GPUs
            gpu_idxs = np.array([gpu.name.split(":")[-1] for gpu in gpu_ids]).astype(
                int
            )

            # Get available memory
            gpu_memories = ProcessPool._get_gpu_memories()[gpu_idxs]

            # Add devices
            devices += [
                Device(id, free_memory, is_gpu=True)
                for id, free_memory in zip(gpu_ids, gpu_memories)
            ]

        else:
            # Run with CPU only
            raise NotImplementedError()

    @staticmethod
    def _get_gpu_memories():
        """
        Function to determine the current available memory of all GPUs on a system.

        Returns
        -------
        gpu_memories: list of int
            Memory of each GPU in MB.
        """
        # Get available GPU memory
        try:
            command = "nvidia-smi --query-gpu=memory.free --format=csv"
            memory_free_info = (
                sp.check_output(command.split()).decode("ascii").split("\n")[:-1][1:]
            )
            gpu_memories = np.array([int(x.split()[0]) for x in memory_free_info])
        except Exception:
            print("Failed to find GPU memory")
            raise RuntimeError(Exception)

        return gpu_memories

    # =======================================================================
    # Getters/Setters

    @property
    def devices(self):
        return self._devices
