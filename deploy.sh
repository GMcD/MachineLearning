#!/bin/bash

make package
git commit -m "Moved VehicleNet to package."
git tag -fa v0.6 -m "VehicleNet v6."
git push
git push origin v0.6
