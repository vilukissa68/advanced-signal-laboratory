#!/usr/bin/env sh


#!/usr/bin/env sh
device="mps"
batch_size=32
epochs=50

t1=$(python ../src/main.py --epochs=$epochs --lr=0.00002 --batch_size=$batch_size \
    --device=$device --optimizer=adamw --silent --train --tensorboard \
    --layers64=3 --layers32=3 --layers16=2 --layers8=1 --dataset="GENKI-4K" --name="adamw")
echo "${t1} 0.00002"

t2=$(python ../src/main.py --epochs=$epochs --lr=0.000002 --batch_size=$batch_size \
    --device=$device --optimizer=adamw --silent --train --tensorboard \
    --layers64=3 --layers32=3 --layers16=2 --layers8=1 --dataset="GENKI-4K" --name="adamw")
echo "${t2} 0.000002"

t3=$(python ../src/main.py --epochs=$epochs --lr=0.0000002 --batch_size=$batch_size \
    --device=$device --optimizer=adamw --silent --train --tensorboard \
    --layers64=3 --layers32=3 --layers16=2 --layers8=1 --dataset="GENKI-4K" --name="adamw")
echo "${t3} 0.0000002"

t4=$(python ../src/main.py --epochs=$epochs --lr=0.00000002 --batch_size=$batch_size \
    --device=$device --optimizer=adamw --silent --train --tensorboard \
    --layers64=3 --layers32=3 --layers16=2 --layers8=1 --dataset="GENKI-4K" --name="adamw")
echo "${t4} 0.00000002"


touch results/adamw2.txt
echo batch_size=$batch_size > results/adamw2.txt
echo epochs=$epochs >> results/adamw2.txt
echo lr=$lr >> results/adamw2.txt

echo "${t1} 0.00002" >> result/adamw2.txt
echo "${t2} 0.000002" >> result/adamw2.txt
echo "${t3} 0.0000002" >> result/adamw2.txt
echo "${t4} 0.00000002" >> result/adamw2.txt
