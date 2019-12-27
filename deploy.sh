#!/bin/bash

make package
git commit -m "Notebook v1."
git tag -a v0.5 -m "Notebook v1."
git push
git push origin v0.5
