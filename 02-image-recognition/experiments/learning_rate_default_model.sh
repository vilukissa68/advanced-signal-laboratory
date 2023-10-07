#!/bin/sh
device="mps"
batch_size=32
epochs=50
t1=$(python ../src/main.py --epochs=$epochs --lr=0.02 --batch_size=$batch_size \
    --device=$device --optimizer=adam --silent)
echo "$t{1} lr=0.02"

t2=$(python ../src/main.py --epochs=$epochs --lr=0.002 --batch_size=$batch_size \
    --device=$device --optimizer=adam --silent)
echo "$t{1} lr=0.002"

t3=$(python ../src/main.py --epochs=$epochs --lr=0.0002 --batch_size=$batch_size \
    --device=$device --optimizer=adam --silent)
echo "$t{1} lr=0.0002"

t4=$(python ../src/main.py --epochs=$epochs --lr=0.0002 --batch_size=$batch_size \
    --device=$device --optimizer=adam --silent)
echo "$t{1} lr=0.00002"

t5=$(python ../src/main.py --epochs=$epochs --lr=0.0002 --batch_size=$batch_size \
    --device=$device --optimizer=adam --silent)
echo "$t{1} lr=0.000002"


echo $t1 | awk -F " " '{print $2}'
echo $t2 | awk -F " " '{print $2}'
echo $t3 | awk -F " " '{print $2}'
echo $t4 | awk -F " " '{print $2}'
echo $t5 | awk -F " " '{print $2}'
