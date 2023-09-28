num_gpus=4
num_jobs=10
dataset=$1
train_epoch=$2
target_count=1  # 最大允许的进程数量
# --ul 0 --ug 1 --uf 0 --d 1 --gre 0 --lre 0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --ul)
            ul="$2"
            shift 2
            ;;
        --ug)
            ug="$2"
            shift 2
            ;;
        --gre)
            gre="$2"
            shift 2
            ;;
        --lre)
            lre="$2"
            shift 2
            ;;
        --upe)
            upe="$2"
            shift 2
            ;;
        --uat)
            uat="$2"
            shift 2
            ;;
        --uf)
            uf="$2"
            shift 2
            ;;
        *)
            # 忽略未知的选项
            shift
            ;;
    esac
done
# patch_len=$6
# 循环遍历每个任务
# 搜learning rate和decay rate
# 96 best sl 576 gre 0 bs 640 lre 0 best lr 5e-5 ims 16
# 192 best sl 576 720 bs 256 ims 32 lr 5e-5
# 336 best sl 640 bs 256 ims 16 lr 0.0001
# 720 sl 720 ims 16 lr 0.0001

# 96 best sl 576 gre 0 bs 640 lre 0 best lr 5e-5 ims 16
# 192 best sl 576 720 bs 256 ims 32 lr 5e-5
# 336 best sl 640 bs 256 ims 16 lr 0.0001
# 720 sl 720 ims 16 lr 0.0001
for bbs in 256; do
    #  relu gelu tanh
    for predl in 96 192 336 720; do
        for dr in 0.99; do
            for nhl in 1; do
                for norm in revin; do
                    for pl in 1; do
                        for stride in 1; do
                            for ims in 16 32; do
                                for nah in 1; do
                                    for d in 1; do
                                        for pembed in 0 1; do
                                            for lr in 0.00001 0.00005; do
                                                for convkernel in 25; do
                                                    for decom in 13 25 51; do
                                            #  linear dynamic
                                                        for sl in 288 576; do
                                                            # 寻找空闲的 GPU
                                                            gpu_found=false
                                                            max_wait=10000  # 最大等待
                                                            target_count=1  # 最大允许的进程数量
                                                            running_processes=0  # 当前正在运行的进程数量
                                                            wait_count=0  # 当前等待次数
                                                            while ! $gpu_found && ((wait_count<max_wait)); do
                                                                for ((gpu=0; gpu<num_gpus; gpu++)); do
                                                                    count=$(nvidia-smi -i $gpu | grep -c "python3")
                                                                    if ((count <= target_count)); then
                                                                        gpu_found=true
                                                                        sleep 12
                                                                        break
                                                                    else
                                                                        wait_count=$((wait_count + 1))
                                                                        sleep 5  # 等待一秒后重新检查
                                                                    fi
                                                                    # if ! nvidia-smi -i $gpu | grep -q "python3"; then
                                                                done
                                                                if ! $gpu_found; then
                                                                    echo "等待空闲的 GPU..."
                                                                    sleep 5  # 等待 10 秒后重新检查 GPU 状态
                                                                    ((wait_count++))
                                                                fi
                                                            done

                                                            # 如果找到空闲的 GPU，则执行任务
                                                            if $gpu_found; then
                                                            python3 -u UniTS_Exp.py --gpu $gpu --dataset $dataset --prel $predl --rs None --epoch $train_epoch --ha relu --lr $lr --pl $pl --dr $dr --bs $bbs --ast 1 --sl $sl --nah $nah --stride $stride --upe $upe --uat $uat --nhl 1 --d $d --pp no --norm revin --ims $ims --pem $pembed --dek $decom --convk $convkernel --ul $ul --ug $ug --uf $uf --d $d --gre $gre --lre $lre &
                                                            else
                                                            echo "无空闲 GPU 可用"
                                                            fi

                                                            # 等待 1 秒，以免在同一 GPU 上同时执行多个任务
                                                            sleep 10
                                                        done
                                                    done
                                                done
                                            done
                                        done
                                    done
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done



