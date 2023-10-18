#!/usr/bin/env sh
device="mps"
batch_size=32
epochs=150
lr=0.0002

t1=$(python ../src/main.py --epochs=$epochs --lr=$lr --batch_size=$batch_size \
    --device=$device --optimizer=adam --silent --train --tensorboard \
    --layers64=2 --layers32=4 --layers16=4 --layers8=1 --dataset="GENKI-4K")
echo "${t1} 2x64 4x64 4x16 1x8 GENKI-4K"

t2=$(python ../src/main.py --epochs=$epochs --lr=$lr --batch_size=$batch_size \
    --device=$device --optimizer=adam --silent --train --tensorboard \
    --layers64=2 --layers32=4 --layers16=4 --layers8=1 --dataset="GENKI-4K-Grayscale")
echo "${t2} 2x64 4x64 4x16 1x8 GENKI-4K-Grayscale"

t3=$(python ../src/main.py --epochs=$epochs --lr=$lr --batch_size=$batch_size \
    --device=$device --optimizer=adam --silent --train --tensorboard \
    --layers64=2 --layers32=4 --layers16=4 --layers8=1 --dataset="GENKI-4K-Augmented")
echo "${t3} 2x64 4x64 4x16 1x8 GENKI-4K-Augmented"

t4=$(python ../src/main.py --epochs=$epochs --lr=$lr --batch_size=$batch_size \
    --device=$device --optimizer=adam --silent --train --tensorboard \
    --layers64=2 --layers32=4 --layers16=4 --layers8=1 --dataset="GENKI-4K-Grayscale-Augmented")
echo "${t4} 2x64 4x64 4x16 1x8 GENKI-4K-Grayscale-Augmented"

touch results/train_best.txt
echo batch_size=$batch_size > results/train_best.txt
echo epochs=$epochs >> results/train_best.txt
echo lr=$lr >> results/train_best.txt

echo "${t1} 3x64 3x64 2x16 1x8 GENKI-4K" >> results/train_best.txt
echo "${t2} 3x64 3x64 2x16 1x8 GENKI-4K-Grayscale" >> results/train_best.txt
echo "${t3} 3x64 3x64 2x16 1x8 GENKI-4K-Augmented" >> results/train_best.txt
echo "${t4} 3x64 3x64 2x16 1x8 GENKI-4K-Grayscale-Augmented" >> results/train_best.txt
