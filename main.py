
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--clearstate", help="ClearState will clear all model and run data if set to True.", action="store_true")
parser.add_argument("--iterations", help="Iterations is the count of training runs of 'samples' executed.", type=int, default=1)
parser.add_argument("--samples", help="Samples is the number of images in each iteration.", type=int, default=2000)
args = parser.parse_args()


from gmcdml.app.vehicle import VehicleNet

if __name__ == '__main__':
    vn = VehicleNet()
    if args.clearstate:
        vn.clearState()
    vn.trainAndReport(iterations=args.iterations, samples=args.samples)
    vn.trainAndReport(iterations=2, samples=args.samples)
    vn.trainAndReport(iterations=1, samples=args.samples)
