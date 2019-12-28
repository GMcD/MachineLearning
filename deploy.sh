#!/bin/bash

if [ $# -lt 2 ]; then
  echo "Usage: ./deply.sh VER MESSAGE..."
  exit 1
fi

VER=$1
MSG=$2

if ! [ $(grep $VER setup.py) ]; then
  echo "Update setup.py with $VER..."
  exit
fi

make package
git add .
git commit -m "${MSG}"
git push origin :refs/tags/v${VER}
git tag -fa v${VER} -m "VehicleNet v${VER}."
git push
git push origin v${VER}
