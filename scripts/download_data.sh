#!/bin/bash
# Download MMEA standard benchmark datasets
# Source: https://github.com/liyichen-cly/MMEA

set -e

DATA_DIR="data"
mkdir -p "$DATA_DIR"

echo "=== Downloading MMEA benchmark datasets ==="
echo "Please download from the official MMEA repository:"
echo "  https://github.com/liyichen-cly/MMEA"
echo ""
echo "Expected layout after download:"
echo "  data/FB15K-DB15K/"
echo "  data/FB15K-YG15K/"
echo ""
echo "Each subdirectory should contain:"
echo "  ent_ids_1, ent_ids_2, rel_ids_1, rel_ids_2"
echo "  triples_1, triples_2"
echo "  train_links, test_links"
echo "  images/  (entity image features, e.g. resnet50 .npy files)"
echo "  entity_descriptions.txt"
echo ""
echo "For MEAformer official preprocessed features:"
echo "  https://github.com/zjukg/MEAformer"
