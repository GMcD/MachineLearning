
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--clearstate", help="ClearState will clear all model and run data if set to True.", action="store_true")
parser.add_argument("--iterations", help="Iterations is the count of training runs of 'samples' executed.", type=int, default=1)
parser.add_argument("--samples", help="Samples is the number of images in each iteration.", type=int, default=2000)
args = parser.parse_args()


from gmcdml.app.vehicle import VehicleNet

if __name__ == '__main__':
    vn = VehicleNet(clearstate=args.clearstate)
    vn.trainAndReport(iterations=args.iterations, samples=args.samples)
