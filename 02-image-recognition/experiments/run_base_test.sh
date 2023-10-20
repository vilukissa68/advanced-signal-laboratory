#!/usr/bin/env sh

device="mps"
batch_size=32
epochs=50
lr=0.00002

t1=$(python ../src/main.py --epochs=$epochs --lr=$lr --batch_size=$batch_size \
    --device=$device --optimizer=adam --silent --train --tensorboard \
    --dataset="GENKI-4K" --name="basic")
echo "1: ${t1}"

t2=$(python ../src/main.py --epochs=$epochs --lr=$lr --batch_size=$batch_size \
    --device=$device --optimizer=adam --silent --train --tensorboard \
    --dataset="GENKI-4K" --name="basic")
echo "2: ${t2}"

t3=$(python ../src/main.py --epochs=$epochs --lr=$lr --batch_size=$batch_size \
    --device=$device --optimizer=adam --silent --train --tensorboard \
    --dataset="GENKI-4K" --name="basic")
echo "3: ${t3}"

t4=$(python ../src/main.py --epochs=$epochs --lr=$lr --batch_size=$batch_size \
    --device=$device --optimizer=adam --silent --train --tensorboard \
    --dataset="GENKI-4K" --name="basic")
echo "4: ${t4}"

t5=$(python ../src/main.py --epochs=$epochs --lr=$lr --batch_size=$batch_size \
    --device=$device --optimizer=adam --silent --train --tensorboard \
    --dataset="GENKI-4K" --name="basic")
echo "5: ${t5}"

touch results/basic_model_test.txt
echo batch_size=$batch_size > results/basic_model_test.txt
echo epochs=$epochs >> results/basic_model_test.txt
echo lr=$lr >> results/basic_model_test.txt

echo "1: ${t1}" >> results/basic_model_test.txt
echo "2: ${t2}" >> results/basic_model_test.txt
echo "3: ${t3}" >> results/basic_model_test.txt
echo "4: ${t4}" >> results/basic_model_test.txt
echo "5: ${t5}" >> results/basic_model_test.txt
