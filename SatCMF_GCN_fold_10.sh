# Define GNN types array
gnn_types=("gcn")

# Define k_hop value range
k_hop_values=(3)

# Iterate over directories
for fold in ./Dataset/10_fold_cv/*; do
    fold_name=$(basename "$fold")
    
    # Continue execution only when the directory name is fold10
    if [ "$fold_name" == "fold10" ]; then
        outdir="./data_analysize/SATCMF_train_result/${fold_name}"
        
        # Iterate over each GNN type
        for gnn_type in "${gnn_types[@]}"; do
            
            # Iterate over each k_hop value
            for k_hop in "${k_hop_values[@]}"; do

                # Run the command and capture the output in real-time
                python ./train.py --gnn-type "$gnn_type" --k-hop "$k_hop" --data-path "$fold" --outdir "$outdir" --cuda-device 0 --seed 158 --use-fp-density-morgan 2>&1 | while IFS= read -r line; do
                    echo "$line"
                done

                # Check the command execution status
                status=$?
                if [ $status -ne 0 ]; then
                    echo "Command execution failed with return code: $status"
                fi
            done
        done
    fi
done
