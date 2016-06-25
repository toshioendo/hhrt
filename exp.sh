#!/bin/sh
# temp command by Endo
make clean
dir=$HOME/export/hhrt
#scp -r . 192.168.8.59:$dir
rm -rf $dir
cp -r -p . $dir
date > $dir/date.txt
echo Written directory $dir
