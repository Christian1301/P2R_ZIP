#!/bin/bash
set -euo pipefail

run_pipeline() {
    local tag="$1"
    local config_path="$2"

    local log_dir="logs/${tag}"
    rm -rf "${log_dir}"
    mkdir -p "${log_dir}"

    echo "[${tag}] Stage 1 → ZIP"
    python3 train_stage1_zip.py --config "${config_path}" > "${log_dir}/stage1.log" 2>&1
    echo "[${tag}] Stage 1 completato"

    echo "[${tag}] Stage 2 → P2R"
    python3 train_stage2_p2r.py --config "${config_path}" > "${log_dir}/stage2.log" 2>&1
    echo "[${tag}] Stage 2 completato"

    echo "[${tag}] Stage 3 → Joint"
    python3 train_stage3_joint.py --config "${config_path}" > "${log_dir}/stage3.log" 2>&1
    echo "[${tag}] Stage 3 completato"

    echo "[${tag}] Eval Stage 1"
    python3 evaluate_stage1.py --config "${config_path}" > "${log_dir}/ev_stage1.log" 2>&1
    echo "[${tag}] Eval Stage 1 completata"

    echo "[${tag}] Eval Stage 2"
    python3 evaluate_stage2.py --config "${config_path}" > "${log_dir}/ev_stage2.log" 2>&1
    echo "[${tag}] Eval Stage 2 completata"

    echo "[${tag}] Eval Stage 3"
    python3 evaluate_stage3.py --config "${config_path}" > "${log_dir}/ev_stage3.log" 2>&1
    echo "[${tag}] Eval Stage 3 completata"

    echo "[${tag}] Visualize gating"
    python3 visualize_gating.py --config "${config_path}" > "${log_dir}/visualize_gating.log" 2>&1 \
        || echo "[${tag}] Visualize fallito (normale se richiede interazione)"
}

# ===========================
#   RUN 1 — CONFIG TESI
# ===========================
run_pipeline "shha_tesi" "config.yaml"

# ===========================
#   RUN 2 — CONFIG BEST
# ===========================
run_pipeline "shha_best" "config_best.yaml"
