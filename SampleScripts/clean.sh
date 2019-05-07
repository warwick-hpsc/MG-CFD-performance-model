#!/bin/bash

set -e
set -u

_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -d "${_dir}"/Prediction ] ; then
	rm -r "${_dir}"/Prediction
fi
if [ -d "${_dir}"/Training ] ; then
	rm -r "${_dir}"/Training
fi