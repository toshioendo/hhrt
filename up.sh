#!/bin/sh
# temp command by Endo
make clean
dir=backup/hhrt-`date +%y%m%d-%H%M`
#scp -r . 192.168.8.59:$dir
cp -r -p . $HOME/$dir
echo Written directory $dir
