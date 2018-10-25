#!/usr/bin/env bash

set -ex
cd ..
$PYTHON setup.py install --single-version-externally-managed --record=record.txt  # Python command to install the script.