#!/bin/bash
dataset=$1
typology="normal"

repetitions=1
sensitivity=500

echo "The chosen dataset is ${dataset}"
echo "The number of repetitions is ${repetitions}"

for ranker in 0 1 2 4 6
do
    for repetition in $(seq ${repetitions})
    do
        case $ranker in
        1)
        #training
        echo "Executing ranker 1"
        time0=$(gdate +%s%N)
        java -jar RankLib-2.12.jar -ranker ${ranker} -train ${typology}/${dataset}/sources_relevance/training${sensitivity}.txt -epoch 50 -layer 2 -save ${typology}/${dataset}/model_${repetition}_${dataset}_ranker${ranker}_${sensitivity}.txt
        time1=$(gdate +%s%N)
        echo $((${time1}-${time0})) >> training_time_ranker${ranker}_${sensitivity}.txt

        #test
        time0=$(gdate +%s%N)
        java -jar RankLib-2.12.jar -load ${typology}/${dataset}/model_${repetition}_${dataset}_ranker${ranker}_${sensitivity}.txt -rank ${typology}/${dataset}/sources_relevance/test${sensitivity}.txt -score ${typology}/${dataset}/${dataset}Score_${repetition}_ranker${ranker}_${sensitivity}.txt
        time1=$(gdate +%s%N)
        echo $((${time1}-${time0})) >> test_time_ranker${ranker}_${sensitivity}.txt
        ;;

        6)
        #training
        echo "Executing ranker 6"
        time0=$(gdate +%s%N)
        java -jar RankLib-2.12.jar -ranker ${ranker} -train ${typology}/${dataset}/sources_relevance/training${sensitivity}.txt -save ${typology}/${dataset}/model_${repetition}_${dataset}_ranker${ranker}_${sensitivity}.txt -tree 30 -metric2T NDCG@10
        time1=$(gdate +%s%N)
        echo $((${time1}-${time0})) >> training_time_ranker${ranker}_${sensitivity}.txt

        #test
        time0=$(gdate +%s%N)
        java -jar RankLib-2.12.jar -load ${typology}/${dataset}/model_${repetition}_${dataset}_ranker${ranker}_${sensitivity}.txt -rank ${typology}/${dataset}/sources_relevance/test${sensitivity}.txt -score ${typology}/${dataset}/${dataset}Score_${repetition}_ranker${ranker}_${sensitivity}.txt -metric2T NDCG@10
        time1=$(gdate +%s%N)
        echo $((${time1}-${time0})) >> test_time_ranker${ranker}_${sensitivity}.txt
        ;;

        *)  
        #training
        time0=$(gdate +%s%N)
        java -jar RankLib-2.12.jar -ranker ${ranker} -train ${typology}/${dataset}/sources_relevance/training${sensitivity}.txt -save ${typology}/${dataset}/model_${repetition}_${dataset}_ranker${ranker}_${sensitivity}.txt
        time1=$(gdate +%s%N)
        echo $((${time1}-${time0})) >> training_time_ranker${ranker}_${sensitivity}.txt


        #test
        time0=$(gdate +%s%N)
        java -jar RankLib-2.12.jar -load ${typology}/${dataset}/model_${repetition}_${dataset}_ranker${ranker}_${sensitivity}.txt -rank ${typology}/${dataset}/sources_relevance/test${sensitivity}.txt -score ${typology}/${dataset}/${dataset}Score_${repetition}_ranker${ranker}_${sensitivity}.txt
        time1=$(gdate +%s%N)
        echo $((${time1}-${time0})) >> test_time_ranker${ranker}_${sensitivity}.txt
        ;;
        esac


        #metrics
        python3 metrics_calculation/metrics_s.py dataset_sources/${typology}/commons_${dataset}_result.csv ${typology}/${dataset}/${dataset}Score_${repetition}_ranker${ranker}_${sensitivity}.txt ${ranker} ${sensitivity}
    done
done

for ranker in 0 1 2 4 6
do
    #python3 metrics_calculation/mean.py _summary_ranker${ranker}.csv 
    #mv mean_summary_ranker${ranker}.csv ${typology}/${dataset}
    mv _summary_ranker${ranker}_${sensitivity}.csv ${typology}/${dataset}
    mv training_time_ranker${ranker}_${sensitivity}.txt ${typology}/${dataset}
    mv test_time_ranker${ranker}_${sensitivity}.txt ${typology}/${dataset}
done
echo "Completed"