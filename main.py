from gmcdml.app.vehicle import VehicleNet

if __name__ == '__main__':
    vn = VehicleNet(clearruns=False)
    vn.trainAndReport(iterations=1)
