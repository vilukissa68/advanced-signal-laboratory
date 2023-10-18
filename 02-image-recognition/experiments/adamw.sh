#!/usr/bin/env sh

#!/usr/bin/env sh
device="mps"
batch_size=32
epochs=50

t1=$(python ../src/main.py --epochs=$epochs --lr=0.2 --batch_size=$batch_size \
    --device=$device --optimizer=adamw --silent --train --tensorboard \
    --layers64=3 --layers32=3 --layers16=2 --layers8=1 --dataset="GENKI-4K"
   --name="adamw")
echo "${t1} 0.2"

t2=$(python ../src/main.py --epochs=$epochs --lr=0.02 --batch_size=$batch_size \
    --device=$device --optimizer=adamw --silent --train --tensorboard \
    --layers64=3 --layers32=3 --layers16=2 --layers8=1 --dataset="GENKI-4K" --name="adamw")
echo "${t2} 0.02"

t3=$(python ../src/main.py --epochs=$epochs --lr=0.002 --batch_size=$batch_size \
    --device=$device --optimizer=adamw --silent --train --tensorboard \
    --layers64=3 --layers32=3 --layers16=2 --layers8=1 --dataset="GENKI-4K" --name="adamw")
echo "${t3} 0.002"

t4=$(python ../src/main.py --epochs=$epochs --lr=0.0002 --batch_size=$batch_size \
    --device=$device --optimizer=adamw --silent --train --tensorboard \
    --layers64=3 --layers32=3 --layers16=2 --layers8=1 --dataset="GENKI-4K" --name="adamw")
echo "${t4} 0.0002"


touch results/adamw.txt
echo batch_size=$batch_size > results/adamw.txt
echo epochs=$epochs >> results/adamw.txt
echo lr=$lr >> results/adamw.txt

echo "${t1} 0.2" >> result/adamw.txt
echo "${t2} 0.02" >> result/adamw.txt
echo "${t3} 0.002" >> result/adamw.txt
echo "${t4} 0.0002" >> result/adamw.txt
