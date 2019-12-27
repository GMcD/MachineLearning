#!/bin/bash

VER=7

make package
git add .
git commit -m "Moved VehicleNet to package."
git push origin :refs/tags/v0.${VER}
git tag -fa v0.${VER} -m "VehicleNet v${VER}."
git push
git push origin v0.${VER}
