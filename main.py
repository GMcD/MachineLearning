from gmcdml.app.vehicle import VehicleNet

if __name__ == '__main__':
    vn = VehicleNet()
    vn.trainAndReport(iterations=1, clearlogs=True)
