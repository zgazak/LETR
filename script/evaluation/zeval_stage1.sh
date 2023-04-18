# Fail the script if there is any failure
set -e

if [[ $# -eq 0 ]] ; then
    echo 'Require Experiment Name'
    exit 1
fi
name=$1
output=exp/$name

epoch=("499")

for ((i=0;i<${#epoch[@]};++i)); do
    mkdir  -p $output/score
    mkdir  -p $output/score/eval-sAP
    mkdir  -p $output/score/eval-fscore
    mkdir  -p $output/score/eval-sAP-york
    mkdir  -p $output/score/eval-fscore-york

    PYTHONPATH=$PYTHONPATH:./src \
    python ./src/main.py --coco_path /data/zgazak/coco/shs_010118-021818_021918-030118 \
    --output_dir $output --backbone resnet101 --resume $output/checkpoints/checkpoint.pth \
    --batch_size 1 ${@:2}  --num_queries 1000  \
    --eval --benchmark --dataset val --append_word ${epoch[i]} --no_aux_loss

    python evaluation/eval-sAP-york.py $output/benchmark/benchmark_york_${epoch[i]} | tee $output/score/eval-sAP-york/${epoch[i]}.txt

    python evaluation/eval-fscore-york.py $output/benchmark/benchmark_york_${epoch[i]} | tee $output/score/eval-fscore-york/${epoch[i]}.txt


done

