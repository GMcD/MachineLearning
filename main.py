
import argparse
parser = argparse.ArgumentParser(description="""
    VehicleNet is a simple implementation of a CNN to classify the CIFAR20 database of images.
    It is based on the `https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html`, but develops that
    sample code with 
        TensorBoard Integration, Persistent Model State, PyPi package allowing pip install, and more.
    This python module provides a simple command line interface to the API, which is also available in 
    the GMcD_ML Jupyter notebook.
""")
parser.add_argument("--clearstate", help="ClearState will clear all model and run data if set to True.", action="store_true", default=False)
parser.add_argument("--iterations", help="Iterations is the count of training runs of 'samples' executed.", type=int, default=1)
parser.add_argument("--samples", help="Samples is the number of images in each iteration.", type=int, default=2000)
args = parser.parse_args()


from gmcdml.app.vehicle import VehicleNet

if __name__ == '__main__':
    vn = VehicleNet()
    if args.clearstate:
        vn.clearState()
    vn.trainAndReport(iterations=args.iterations, samples=args.samples)
