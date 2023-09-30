#!/bin/bash
parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
cd "$parent_path"
cd ../data
wget https://inc.ucsd.edu/~nick/GENKI-R2009a.tgz
tar zxvf GENKI-R2009a.tgz
rm GENKI-R2009a.tgz
