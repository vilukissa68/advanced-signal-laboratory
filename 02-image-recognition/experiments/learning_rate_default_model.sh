#!/bin/sh
device="mps"
batch_size=32
epochs=50
t1=$(python ../src/main.py --epochs=$epochs --lr=0.2 --batch_size=$batch_size \
    --device=$device --optimizer=adam --silent --train --tensorboard \
    --layers64=3 --layers32=3 --layers16=2 --layers8=1 --dataset="GENKI-4K" --name="adam")
echo "${t1} 0.2"

t2=$(python ../src/main.py --epochs=$epochs --lr=0.02 --batch_size=$batch_size \
    --device=$device --optimizer=adam --silent --train --tensorboard \
    --layers64=3 --layers32=3 --layers16=2 --layers8=1 --dataset="GENKI-4K" --name="adam")
echo "${t2} 0.02"

t3=$(python ../src/main.py --epochs=$epochs --lr=0.002 --batch_size=$batch_size \
    --device=$device --optimizer=adam --silent --train --tensorboard \
    --layers64=3 --layers32=3 --layers16=2 --layers8=1 --dataset="GENKI-4K" --name="adam")
echo "${t3} 0.002"

t4=$(python ../src/main.py --epochs=$epochs --lr=0.0002 --batch_size=$batch_size \
    --device=$device --optimizer=adam --silent --train --tensorboard \
    --layers64=3 --layers32=3 --layers16=2 --layers8=1 --dataset="GENKI-4K" --name="adam")
echo "${t4} 0.0002"

t5=$(python ../src/main.py --epochs=$epochs --lr=0.00002 --batch_size=$batch_size \
    --device=$device --optimizer=adam --silent --train --tensorboard \
    --layers64=3 --layers32=3 --layers16=2 --layers8=1 --dataset="GENKI-4K" --name="adam")
echo "${t1} 0.00002"

t6=$(python ../src/main.py --epochs=$epochs --lr=0.000002 --batch_size=$batch_size \
    --device=$device --optimizer=adam --silent --train --tensorboard \
    --layers64=3 --layers32=3 --layers16=2 --layers8=1 --dataset="GENKI-4K" --name="adam")
echo "${t2} 0.000002"

t7=$(python ../src/main.py --epochs=$epochs --lr=0.0000002 --batch_size=$batch_size \
    --device=$device --optimizer=adam --silent --train --tensorboard \
    --layers64=3 --layers32=3 --layers16=2 --layers8=1 --dataset="GENKI-4K" --name="adam")
echo "${t3} 0.0000002"

t8=$(python ../src/main.py --epochs=$epochs --lr=0.00000002 --batch_size=$batch_size \
    --device=$device --optimizer=adam --silent --train --tensorboard \
    --layers64=3 --layers32=3 --layers16=2 --layers8=1 --dataset="GENKI-4K" --name="adam")
echo "${t4} 0.00000002"


touch results/learning_rates.txt
echo batch_size=$batch_size > results/learning_rates.txt
echo epochs=$epochs >> results/learning_rates.txt

echo "${t1} 0.2" >> results/learning_rates.txt
echo "${t2} 0.02" >> results/learning_rates.txt
echo "${t3} 0.002" >> results/learning_rates.txt
echo "${t4} 0.0002" >> results/learning_rates.txt
echo "${t5} 0.00002" >> results/learning_rates.txt
echo "${t6} 0.000002" >> results/learning_rates.txt
echo "${t7} 0.0000002" >> results/learning_rates.txt
echo "${t8} 0.00000002" >> results/learning_rates.txt
