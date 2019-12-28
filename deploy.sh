#!/bin/bash

if [ $# -lt 2 ]; then
  echo "Usage: ./deply.sh VER MESSAGE..."
  exit 1
fi

VER=$1
MSG=$2

make package
git add .
git commit -m "${MSG}"
git push origin :refs/tags/v0.${VER}
git tag -fa v0.${VER} -m "VehicleNet v${VER}."
git push
git push origin v0.${VER}
