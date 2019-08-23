#!/bin/bash
dataset=$1
typology="normal"

repetitions=30

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
        java -jar RankLib-2.12.jar -ranker ${ranker} -train ${typology}/${dataset}/sources_relevance/shuffled/training${repetitions}.txt -epoch 50 -layer 2 -save ${typology}/${dataset}/model_${repetition}_${dataset}_ranker${ranker}.txt
        time1=$(gdate +%s%N)
        echo $((${time1}-${time0})) >> training_time_ranker${ranker}.txt

        #test
        time0=$(gdate +%s%N)
        java -jar RankLib-2.12.jar -load ${typology}/${dataset}/model_${repetition}_${dataset}_ranker${ranker}.txt -rank ${typology}/${dataset}/sources_relevance/test.txt -score ${typology}/${dataset}/${dataset}Score_${repetition}_ranker${ranker}.txt
        time1=$(gdate +%s%N)
        echo $((${time1}-${time0})) >> test_time_ranker${ranker}.txt
        ;;

        6)
        #training
        echo "Executing ranker 6"
        time0=$(gdate +%s%N)
        java -jar RankLib-2.12.jar -ranker ${ranker} -train ${typology}/${dataset}/sources_relevance/shuffled/training${repetitions}.txt -save ${typology}/${dataset}/model_${repetition}_${dataset}_ranker${ranker}.txt -tree 30 -metric2T NDCG@10
        time1=$(gdate +%s%N)
        echo $((${time1}-${time0})) >> training_time_ranker${ranker}.txt

        #test
        time0=$(gdate +%s%N)
        java -jar RankLib-2.12.jar -load ${typology}/${dataset}/model_${repetition}_${dataset}_ranker${ranker}.txt -rank ${typology}/${dataset}/sources_relevance/test.txt -score ${typology}/${dataset}/${dataset}Score_${repetition}_ranker${ranker}.txt -metric2T NDCG@10
        time1=$(gdate +%s%N)
        echo $((${time1}-${time0})) >> test_time_ranker${ranker}.txt
        ;;

        *)  
        #training
        time0=$(gdate +%s%N)
        java -jar RankLib-2.12.jar -ranker ${ranker} -train ${typology}/${dataset}/sources_relevance/shuffled/training${repetitions}.txt -save ${typology}/${dataset}/model_${repetition}_${dataset}_ranker${ranker}.txt
        time1=$(gdate +%s%N)
        echo $((${time1}-${time0})) >> training_time_ranker${ranker}.txt


        #test
        time0=$(gdate +%s%N)
        java -jar RankLib-2.12.jar -load ${typology}/${dataset}/model_${repetition}_${dataset}_ranker${ranker}.txt -rank ${typology}/${dataset}/sources_relevance/test.txt -score ${typology}/${dataset}/${dataset}Score_${repetition}_ranker${ranker}.txt
        time1=$(gdate +%s%N)
        echo $((${time1}-${time0})) >> test_time_ranker${ranker}.txt
        ;;
        esac


        #metrics
        python3 metrics_calculation/metrics.py dataset_sources/${typology}/commons_${dataset}_result.csv ${typology}/${dataset}/${dataset}Score_${repetition}_ranker${ranker}.txt ${ranker}
    done
done

for ranker in 0 1 2 4 6
do
    python3 metrics_calculation/mean.py _summary_ranker${ranker}.csv 
    mv mean_summary_ranker${ranker}.csv ${typology}/${dataset}
    mv _summary_ranker${ranker}.csv ${typology}/${dataset}
    mv training_time_ranker${ranker}.txt ${typology}/${dataset}
    mv test_time_ranker${ranker}.txt ${typology}/${dataset}
done
echo "Completed"