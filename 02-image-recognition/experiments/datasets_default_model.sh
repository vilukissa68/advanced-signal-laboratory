#!/usr/bin/env sh
device="mps"
batch_size=32
epochs=50
lr=0.0002

t1=$(python ../src/main.py --epochs=$epochs --lr=$lr --batch_size=$batch_size \
    --device=$device --optimizer=adam --silent --name="datasets" \
    --train --dataset="GENKI-4K" --tensorboard)
echo "GENKI-4K: ${t1}"

t2=$(python ../src/main.py --epochs=$epochs --lr=$lr --batch_size=$batch_size \
    --device=$device --optimizer=adam --silent --name="datasets" \
    --train --dataset="GENKI-4K-Grayscale" --tensorboard)
echo "GENKI-4K-Grayscale: ${t2}"

t3=$(python ../src/main.py --epochs=$epochs --lr=$lr --batch_size=$batch_size \
    --device=$device --optimizer=adam --silent --name="datasets" \
    --train --dataset="GENKI-4K-Augmented" --tensorboard)
echo "GENKI-4K-Augmented: ${t3}"

t4=$(python ../src/main.py --epochs=$epochs --lr=$lr --batch_size=$batch_size \
    --device=$device --optimizer=adam --silent --name="datasets" \
    --train --dataset="GENKI-4K-Grayscale-Augmented" --tensorboard)
echo "GENKI-4K-Grayscale-Augmented: ${t4}"

# Write to file
touch results/data_default_model.txt
echo batch_size=$batch_size > results/data_default_model.txt
echo epochs=$epochs >> results/data_default_model.txt
echo lr=$lr >> results/data_default_model.txt

echo "GENKI-4K: ${t1}" >> results/datasets_default_model.txt
echo "GENKI-4K-Grayscale: ${t2}" >> results/datasets_default_model.txt
echo "GENKI-4K-Augmented: ${t3}" >> results/datasets_default_model.txt
echo "GENKI-4K-Grayscale-Augmented: ${t4}" >> results/datasets_default_model.txt
