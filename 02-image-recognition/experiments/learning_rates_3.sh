#!/usr/bin/env sh

#!/usr/bin/env sh


#!/usr/bin/env sh
device="mps"
batch_size=32
epochs=50

t1=$(python ../src/main.py --epochs=$epochs --lr=0.00005 --batch_size=$batch_size \
    --device=$device --optimizer=adam --silent --train --tensorboard \
    --layers64=3 --layers32=3 --layers16=2 --layers8=1 --dataset="GENKI-4K" --name="adam")
echo "${t1} Adam 0.00005"

t2=$(python ../src/main.py --epochs=$epochs --lr=0.0001 --batch_size=$batch_size \
    --device=$device --optimizer=adam --silent --train --tensorboard \
    --layers64=3 --layers32=3 --layers16=2 --layers8=1 --dataset="GENKI-4K" --name="adam")
echo "${t2} Adam 0.0001"

t3=$(python ../src/main.py --epochs=$epochs --lr=0.0003 --batch_size=$batch_size \
    --device=$device --optimizer=adam --silent --train --tensorboard \
    --layers64=3 --layers32=3 --layers16=2 --layers8=1 --dataset="GENKI-4K" --name="adam")
echo "${t3} Adam 0.00013"

t4=$(python ../src/main.py --epochs=$epochs --lr=0.0005 --batch_size=$batch_size \
    --device=$device --optimizer=adam --silent --train --tensorboard \
    --layers64=3 --layers32=3 --layers16=2 --layers8=1 --dataset="GENKI-4K" --name="adam")
echo "${t4} Adam 0.0005"

t5=$(python ../src/main.py --epochs=$epochs --lr=0.0007 --batch_size=$batch_size \
    --device=$device --optimizer=adam --silent --train --tensorboard \
    --layers64=3 --layers32=3 --layers16=2 --layers8=1 --dataset="GENKI-4K" --name="adam")
echo "${t5} Adam 0.0007"

# AdamW
t6=$(python ../src/main.py --epochs=$epochs --lr=0.00005 --batch_size=$batch_size \
    --device=$device --optimizer=adamw --silent --train --tensorboard \
    --layers64=3 --layers32=3 --layers16=2 --layers8=1 --dataset="GENKI-4K" --name="adamw")
echo "${t6} Adam 0.00005"

t7=$(python ../src/main.py --epochs=$epochs --lr=0.0001 --batch_size=$batch_size \
    --device=$device --optimizer=adamw --silent --train --tensorboard \
    --layers64=3 --layers32=3 --layers16=2 --layers8=1 --dataset="GENKI-4K" --name="adamw")
echo "${t7} Adam 0.0001"

t8=$(python ../src/main.py --epochs=$epochs --lr=0.0003 --batch_size=$batch_size \
    --device=$device --optimizer=adamw --silent --train --tensorboard \
    --layers64=3 --layers32=3 --layers16=2 --layers8=1 --dataset="GENKI-4K" --name="adamw")
echo "${t8} Adam 0.00013"

t9=$(python ../src/main.py --epochs=$epochs --lr=0.0005 --batch_size=$batch_size \
    --device=$device --optimizer=adamw --silent --train --tensorboard \
    --layers64=3 --layers32=3 --layers16=2 --layers8=1 --dataset="GENKI-4K" --name="adamw")
echo "${t9} Adam 0.0005"

t10=$(python ../src/main.py --epochs=$epochs --lr=0.0007 --batch_size=$batch_size \
    --device=$device --optimizer=adamw --silent --train --tensorboard \
    --layers64=3 --layers32=3 --layers16=2 --layers8=1 --dataset="GENKI-4K" --name="adamw")
echo "${t10} Adam 0.0007"
