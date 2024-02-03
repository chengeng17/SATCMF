# Define an array of GNN types
gnn_types=("gcn" "gin" "graphsage" "mpnn")

# Define the range of k_hop values
k_hop_values=(1 2 3)

# Iterate over directories
for fold in ./dataset/10_fold_cv/*; do
    fold_name=$(basename "$fold")
    outdir="./ablation_study/without_morgan_fingerprint/ablation_study_result143/${fold_name}"
    
    # Iterate over each GNN type
    for gnn_type in "${gnn_types[@]}"; do
        
        # Iterate over each k_hop value
        for k_hop in "${k_hop_values[@]}"; do

            # Run the command and capture the output in real-time
            python ./train.py --gnn-type "$gnn_type" --k-hop "$k_hop" --data-path "$fold" --outdir "$outdir" --cuda-device 0 --seed 142 2>&1 | while IFS= read -r line; do
                echo "$line"
            done

            # Check the command execution status
            status=$?
            if [ $status -ne 0 ]; then
                echo "Command execution failed with return code: $status"
            fi
        done
    done
done
