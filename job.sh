#!/bin/bash

TRAIN_DIR=runtime

if [ $# -lt 1 ]
then
    echo ""
    echo ""
    echo "****************************** Usage *************************************"
    echo "Create a job as ps(id), worker(id) or single "
    echo "    sh "$0" create [JOB_TYPE(ps/worker/single)] [job_idx] [gpu]"
    echo ""
    echo "Stop all"
    echo "    sh "$0" stop"
    echo "************************************************************"
    echo ""
    exit
fi

if [ "$1"x = "create"x ]
then
    JOB_TYPE=$2
    INDEX=$3
    GPU=$4
    PORT=$5
    PS=$6
    WORKER=$7

    [ ! -d "$TRAIN_DIR" ] && mkdir -p "$TRAIN_DIR"

    if [ "$JOB_TYPEx = "ps"x ] || [ "$JOB_TYPE"x = "worker"x ] || [ "$JOB_TYPE"x = "single"x ]
    then
        touch $TRAIN_DIR/run.pid
        EXISTS=`awk -v job_type=$JOB_TYPE-v idx=$INDEX '{if($1==job_type"_"idx)print}' $TRAIN_DIR/run.pid`
        if [ ! -z $EXISTS ]
        then
           echo -e "\nAready Runing: \n"$EXISTS"\n"
           exit
        fi

        touch $TRAIN_DIR/"train_"$JOB_TYPE"_"$INDEX".log"
        touch $TRAIN_DIR/"std_"$JOB_TYPE"_"$INDEX".log"
        CUDA_VISIBLE_DEVICES=$GPU nohup python -u train.py --job_type=$JOB_TYPE --task_index=$INDEX --gpu=$GPU --data_stream_port=$PORT --ps_hosts=$PS --worker_hosts=$WORKER 2>&1 > $TRAIN_DIR/"std_"$JOB_TYPE"_"$INDEX".log" &

        TIME=`date "+%Y-%m-%d-%T"`
        TMP=`echo $JOB_TYPE"_"$INDEX" "$!" "$DATA_DIR" "$TIME`
        echo $TMP  >> $TRAIN_DIR/status

    else
        echo "Create a job as ps(id), worker(id) or single "
        echo "    sh "$0" create [JOB_TYPE(ps/worker/single)] [job_idx] [gpu]"
        exit
    fi

elif [ "$1"x = "stop"x ]
then
    awk '{print $2}' $TRAIN_DIR//status | xargs -t -I {} kill -9 {}
    sleep 1
    rm $TRAIN_DIR//status
    fi