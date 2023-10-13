#!/bin/sh
device="mps"
batch_size=32
epochs=150
lr=0.0002
dataset="GENKI-4K-Augmented"
t1=$(python ../src/main.py --epochs=$epochs --lr=$lr --batch_size=$batch_size \
    --device=$device --optimizer=adam --silent --train --tensorboard \
    --layers64=2 --layers32=3 --layers16=3 --layers8=2 --dataset=$dataset)
echo "${t1} 2x64 3x64 3x16 2x8"

t2=$(python ../src/main.py --epochs=$epochs --lr=$lr --batch_size=$batch_size \
    --device=$device --optimizer=adam --silent --train --tensorboard \
    --layers64=3 --layers32=4 --layers16=4 --layers8=3 --dataset=$dataset)
echo "${t2} 3x64 4x64 4x16 3x8"

t3=$(python ../src/main.py --epochs=$epochs --lr=$lr --batch_size=$batch_size \
    --device=$device --optimizer=adam --silent --train --tensorboard \
    --layers64=4 --layers32=5 --layers16=5 --layers8=4 --dataset=$dataset)
echo "${t3} 4x64 5x64 5x16 4x8"

t4=$(python ../src/main.py --epochs=$epochs --lr=$lr --batch_size=$batch_size \
    --device=$device --optimizer=adam --silent --train --tensorboard \
    --layers64=2 --layers32=2 --layers16=2 --layers8=1 --dataset=$dataset)
echo "${t4} 2x64 2x64 2x16 1x8"

t5=$(python ../src/main.py --epochs=$epochs --lr=$lr --batch_size=$batch_size \
    --device=$device --optimizer=adam --silent --train --tensorboard \
    --layers64=3 --layers32=2 --layers16=2 --layers8=1 --dataset=$dataset)
echo "${t5} 3x64 2x64 2x16 1x8"

t6=$(python ../src/main.py --epochs=$epochs --lr=$lr --batch_size=$batch_size \
    --device=$device --optimizer=adam --silent --train --tensorboard \
    --layers64=1 --layers32=4 --layers16=4 --layers8=1 --dataset=$dataset)
echo "${t6} 1x64 4x64 4x16 1x8"

t7=$(python ../src/main.py --epochs=$epochs --lr=$lr --batch_size=$batch_size \
    --device=$device --optimizer=adam --silent --train --tensorboard \
    --layers64=3 --layers32=3 --layers16=2 --layers8=1 --dataset=$dataset)
echo "${t7} 3x64 3x64 2x16 1x8"

echo $t1 | awk -F " " '{print $2}'
echo $t2 | awk -F " " '{print $2}'
echo $t3 | awk -F " " '{print $2}'
echo $t4 | awk -F " " '{print $2}'
echo $t5 | awk -F " " '{print $2}'
echo $t6 | awk -F " " '{print $2}'
echo $t7 | awk -F " " '{print $2}'

# Write to file
touch results/extra_layers_default_model.txt
echo batch_size=$batch_size > results/extra_layers_default_model.txt
echo epochs=$epochs >> results/extra_layers_default_model.txt
echo lr=$lr >> results/extra_layers_default_model.txt

echo "${t1} 2x64 3x64 3x16 2x8" >> results/extra_layers_default_model.txt
echo "${t2} 3x64 4x64 4x16 3x8" >> results/extra_layers_default_model.txt
echo "${t3} 4x64 5x64 5x16 4x8" >> results/extra_layers_default_model.txt
echo "${t4} 2x64 2x64 2x16 1x8" >> results/extra_layers_default_model.txt
echo "${t5} 3x64 2x64 2x16 1x8" >> results/extra_layers_default_model.txt
echo "${t6} 1x64 4x64 4x16 1x8" >> results/extra_layers_default_model.txt
echo "${t7} 3x64 3x64 2x16 1x8" >> results/extra_layers_default_model.txt
