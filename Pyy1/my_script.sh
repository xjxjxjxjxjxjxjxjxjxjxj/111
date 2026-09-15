#!/bin/bash

LOG=/home/jetsonjetson/WorkSpace2026/pyy627/Pyy1/start_log/boot.log
INFER_LOG=/home/jetsonjetson/WorkSpace2026/pyy627/Pyy1/start_log/infer.log
CAR_LOG=/home/jetsonjetson/WorkSpace2026/pyy627/Pyy1/start_log/car.log

#################################
# 清理旧日志
#################################

> $LOG
> $INFER_LOG
> $CAR_LOG


echo "=============================" >> $LOG
echo "启动脚本执行时间: $(date)" >> $LOG


#################################
# 清理旧 Python 进程
#################################

echo "清理jetson用户旧Python进程..." >> $LOG


OLD_PIDS=$(ps -u jetsonjetson -f | grep python3 | grep -v grep | awk '{print $2}')


for pid in $OLD_PIDS
do
    echo "kill python PID=$pid" >> $LOG
    kill -9 $pid
done


sleep 3


#################################
# 清理旧infer状态
#################################

rm -f /tmp/infer_ready


#################################
# 启动 infer_back_end.py
#################################

echo "启动 infer_back_end.py..." >> $LOG


python3 "/home/jetsonjetson/WorkSpace2026/pyy627/Pyy1/smartcar/paddlebaidu/infer_cs/base/infer_back_end.py" \
>> $INFER_LOG 2>&1 &


INFER_PID=$!


echo "infer PID=$INFER_PID" >> $LOG


#################################
# 等待 infer 初始化完成
#################################

echo "等待 infer init ok..." >> $LOG


TIMEOUT=120
COUNT=0


while [ ! -f /tmp/infer_ready ]
do
    sleep 1

    COUNT=$((COUNT+1))


    if [ $COUNT -ge $TIMEOUT ]
    then
        echo "infer启动超时!" >> $LOG
        exit 1
    fi

done


echo "infer初始化完成: $(date)" >> $LOG



#################################
# 等待 USB 串口
#################################

echo "等待USB串口 /dev/ttyUSB0..." >> $LOG


USB_TIMEOUT=60
USB_COUNT=0


while [ ! -e /dev/ttyUSB0 ]
do
    echo "$(date): 未检测到 /dev/ttyUSB0" >> $LOG

    sleep 2

    USB_COUNT=$((USB_COUNT+2))


    if [ $USB_COUNT -ge $USB_TIMEOUT ]
    then
        echo "等待USB串口超时!" >> $LOG
        exit 1
    fi

done


echo "/dev/ttyUSB0 已连接" >> $LOG


# 给控制器额外初始化时间
sleep 5


#################################
# 启动小车主程序
#################################

echo "启动小车控制程序..." >> $LOG

sleep 30
cd /home/jetsonjetson/WorkSpace2026/pyy627/Pyy1/
python3 "/home/jetsonjetson/WorkSpace2026/pyy627/Pyy1/car_task_function.py" \
>> $CAR_LOG 2>&1


echo "小车程序退出: $(date)" >> $LOG