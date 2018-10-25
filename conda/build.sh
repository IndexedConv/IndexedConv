#!/usr/bin/env bash

set -ex
cd ..
pwd
$PYTHON setup.py install --single-version-externally-managed --record=record.txt  # Python command to install the script.