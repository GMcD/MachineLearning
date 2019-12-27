#!/bin/bash

make package
git add .
git commit -m "Moved VehicleNet to package."
git push origin :refs/tags/v0.6
git tag -fa v0.6 -m "VehicleNet v6."
git push
git push origin v0.6
