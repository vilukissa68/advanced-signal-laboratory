#!/usr/bin/env sh
#
device="mps"
epochs=50
lr=0.00002

t1=$(python ../src/main.py --epochs=$epochs --lr=$lr --batch_size=1 \
    --device=$device --optimizer=adam --silent --train --tensorboard)
echo "${t1} bs=1"

t2=$(python ../src/main.py --epochs=$epochs --lr=$lr --batch_size=8 \
    --device=$device --optimizer=adam --silent --train --tensorboard)
echo "${t2} bs=8"

t3=$(python ../src/main.py --epochs=$epochs --lr=$lr --batch_size=16 \
    --device=$device --optimizer=adam --silent --train --tensorboard)
echo "${t3} bs=16"

t4=$(python ../src/main.py --epochs=$epochs --lr=$lr --batch_size=32 \
    --device=$device --optimizer=adam --silent --train --tensorboard)
echo "${t4} bs=32"

t5=$(python ../src/main.py --epochs=$epochs --lr=$lr --batch_size=64 \
    --device=$device --optimizer=adam --silent --train --tensorboard)
echo "${t5} bs=64"

t6=$(python ../src/main.py --epochs=$epochs --lr=$lr --batch_size=128 \
    --device=$device --optimizer=adam --silent --train --tensorboard)
echo "${t6} bs=128"

t7=$(python ../src/main.py --epochs=$epochs --lr=$lr --batch_size=256 \
    --device=$device --optimizer=adam --silent --train --tensorboard)
echo "${t7} bs=256"

touch results/batchsizes.txt
echo epochs=$epochs > results/batchsizes.txt
echo lr=$lr >> results/batchsizes.txt

echo "${t1} bs=1" >> results/batchsizes.txt
echo "${t2} bs=8" >> results/batchsizes.txt
echo "${t3} bs=16" >> results/batchsizes.txt
echo "${t4} bs=32" >> results/batchsizes.txt
echo "${t5} bs=64" >> results/batchsizes.txt
echo "${t6} bs=128" >> results/batchsizes.txt
echo "${t7} bs=256" >> results/batchsizes.txt
