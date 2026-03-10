#!/bin/bash

set +e  # Do NOT exit on error

CONFIGS=(
    "gridfm01"
    "gridfm02"
)

CONFIG_PATHS=(
    "../examples/config/gridFMv0.1_pretraining.yaml"
    "../examples/config/gridFMv0.2_pretraining.yaml"
)

GRAPH_SIZES=(
    "30 110"
    "300 1120"
    "2000 9276"
    "3022 11390"
    "9241 41337"
    "30000 100784"
)

OUTPUT_DIR="benchmark_results"
mkdir -p $OUTPUT_DIR
for i in "${!CONFIGS[@]}"; do
    config_name="${CONFIGS[$i]}"
    config_path="${CONFIG_PATHS[$i]}"
    for size in "${GRAPH_SIZES[@]}"; do
        read -r nodes edges <<< "$size"
        output_file="${OUTPUT_DIR}/${config_name}_${nodes}nodes_${edges}edges.csv"
        echo "Running benchmark for $config_name with $nodes nodes and $edges edges..."
        python benchmark_model_inference.py \
            --config "$config_path" \
            --output_csv "$output_file" \
            --num_nodes "$nodes" \
            --num_edges "$edges" || echo "Failed for $config_name with $nodes nodes"
    done
done