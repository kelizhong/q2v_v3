#!/bin/bash

PS='localhost:3221,localhost:3222,localhost:3223,localhost:3224,localhost:3225,localhost:3226,localhost:3227,localhost:3228'
WORKER='localhost:2221,localhost:2222,localhost:2223,localhost:2224,localhost:2225,localhost:2226,localhost:2227,localhost:2228,localhost:4221,localhost:4222,localhost:4223,localhost:4224,localhost:4225,localhost:4226,localhost:4227,localhost:4228'
PORT=5558

index=0
for host in `echo "$PS" | sed 's/,/\n/g'`
do
    echo "Starting ps $host $index"
    sh job.sh create ps $index '' $PORT $PS $WORKER
    index=`expr $index + 1`
done

index=0
for host in `echo "$WORKER" | sed 's/,/\n/g'`
do
    echo "Starting worker $host $index"
    sh job.sh create worker $index $index $PORT $PS $WORKER
    index=`expr $index + 1`;
done
