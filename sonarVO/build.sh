#!/bin/bash

echo "Configuring..."
echo "cmake .."
mkdir ./build -p
mkdir ./bin -p
pushd build >& /dev/null
cmake ..
echo "Building..."
echo "make $@"
make $@
popd >& /dev/null

