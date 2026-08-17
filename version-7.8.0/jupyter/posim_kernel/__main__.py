from ipykernel.kernelapp import IPKernelApp

from .kernel import PosimKernel

IPKernelApp.launch_instance(kernel_class=PosimKernel)
