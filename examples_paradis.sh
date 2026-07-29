#!/bin/bash

##############################################################################
# Patho-Ensemble Examples with PARADIS Data
# Ready-to-use command examples for common workflows
##############################################################################

set -e  # Exit on error

# Color codes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Patho-Ensemble PARADIS Examples${NC}"
echo -e "${BLUE}========================================${NC}\n"

# Configuration
WORK_DIR="/shared/home/PARADIS/datos"
EPOCHS=100

##############################################################################
# EXAMPLE 1: Single Model Training
##############################################################################
example_single_model() {
    echo -e "\n${GREEN}EXAMPLE 1: Single Model Training${NC}"
    echo "Train a single ABMIL model on TP53 mutation in CPTAC-BRCA"
    echo ""
    echo "python src/train_abmil.py \\"
    echo "    --foundational_model ctranspath \\"
    echo "    --latent_dim 768 \\"
    echo "    --work_dir ${WORK_DIR} \\"
    echo "    --train_source cptac_brca \\"
    echo "    --tissue_patching 20x_224px_0px_overlap \\"
    echo "    --task_name TP53_mutation \\"
    echo "    --epochs ${EPOCHS} \\"
    echo "    --fold all"
    echo ""
}

##############################################################################
# EXAMPLE 2: Multi-Model Training (Sequential)
##############################################################################
example_multi_model_seq() {
    echo -e "\n${GREEN}EXAMPLE 2: Multi-Model Training (Sequential)${NC}"
    echo "Train multiple models on the same task"
    echo ""
    echo "for model in ctranspath uni_v2 virchow_v1; do"
    echo "  latent_dim=768"
    echo "  [[ \$model == 'virchow_v1' ]] && latent_dim=2560"
    echo ""
    echo "  python src/train_abmil.py \\"
    echo "    --foundational_model \$model \\"
    echo "    --latent_dim \$latent_dim \\"
    echo "    --work_dir ${WORK_DIR} \\"
    echo "    --train_source cptac_brca \\"
    echo "    --tissue_patching 20x_224px_0px_overlap \\"
    echo "    --task_name TP53_mutation \\"
    echo "    --epochs ${EPOCHS} \\"
    echo "    --fold all"
    echo "done"
    echo ""
}

##############################################################################
# EXAMPLE 3: Multi-Model Training (Parallel)
##############################################################################
example_multi_model_parallel() {
    echo -e "\n${GREEN}EXAMPLE 3: Multi-Model Training (Parallel)${NC}"
    echo "Train multiple models in parallel (recommended for GPU cluster)"
    echo ""
    echo "python src/train_abmil.py --foundational_model ctranspath --latent_dim 768 --work_dir ${WORK_DIR} --train_source cptac_brca --tissue_patching 20x_224px_0px_overlap --task_name TP53_mutation --epochs ${EPOCHS} --fold all &"
    echo "python src/train_abmil.py --foundational_model uni_v2 --latent_dim 768 --work_dir ${WORK_DIR} --train_source cptac_brca --tissue_patching 20x_224px_0px_overlap --task_name TP53_mutation --epochs ${EPOCHS} --fold all &"
    echo "python src/train_abmil.py --foundational_model virchow_v1 --latent_dim 2560 --work_dir ${WORK_DIR} --train_source cptac_brca --tissue_patching 20x_224px_0px_overlap --task_name TP53_mutation --epochs ${EPOCHS} --fold all &"
    echo "wait"
    echo ""
}

##############################################################################
# EXAMPLE 4: Simple Weighted Ensemble
##############################################################################
example_weighted_ensemble() {
    echo -e "\n${GREEN}EXAMPLE 4: Simple Weighted Ensemble${NC}"
    echo "Evaluate ensemble using weighted averaging (after training models)"
    echo ""
    echo "python src/ensemble.py \\"
    echo "    --foundational_models ctranspath uni_v2 virchow_v1 \\"
    echo "    --work_dir ${WORK_DIR} \\"
    echo "    --train_source cptac_brca \\"
    echo "    --tissue_patching 20x_224px_0px_overlap \\"
    echo "    --task_name TP53_mutation \\"
    echo "    --weights_type auc_roc"
    echo ""
}

##############################################################################
# EXAMPLE 5: Meta-Learner Ensemble
##############################################################################
example_meta_learner() {
    echo -e "\n${GREEN}EXAMPLE 5: Meta-Learner Ensemble (Complete Workflow)${NC}"
    echo "Train base models, get training predictions, then train meta-learner"
    echo ""
    echo "# Step 1: Train base models"
    echo "for model in ctranspath uni_v2 virchow_v1; do"
    echo "  latent_dim=768"
    echo "  [[ \$model == 'virchow_v1' ]] && latent_dim=2560"
    echo "  python src/train_abmil.py --foundational_model \$model --latent_dim \$latent_dim --work_dir ${WORK_DIR} --train_source cptac_brca --tissue_patching 20x_224px_0px_overlap --task_name TP53_mutation --epochs ${EPOCHS} --fold all"
    echo "done"
    echo ""
    echo "# Step 2: Get training split predictions"
    echo "for model in ctranspath uni_v2 virchow_v1; do"
    echo "  latent_dim=768"
    echo "  [[ \$model == 'virchow_v1' ]] && latent_dim=2560"
    echo "  python src/test_abmil.py --foundational_model \$model --latent_dim \$latent_dim --work_dir ${WORK_DIR} --train_source cptac_brca --tissue_patching 20x_224px_0px_overlap --task_name TP53_mutation --epochs ${EPOCHS}"
    echo "done"
    echo ""
    echo "# Step 3: Train meta-learner"
    echo "python src/ensemble4.py \\"
    echo "    --foundational_models ctranspath uni_v2 virchow_v1 \\"
    echo "    --work_dir ${WORK_DIR} \\"
    echo "    --train_source cptac_brca \\"
    echo "    --tissue_patching 20x_224px_0px_overlap \\"
    echo "    --task_name TP53_mutation"
    echo ""
}

##############################################################################
# EXAMPLE 6: Multiple Tasks
##############################################################################
example_multiple_tasks() {
    echo -e "\n${GREEN}EXAMPLE 6: Multiple Tasks on Same Dataset${NC}"
    echo "Train models for multiple prediction targets in parallel"
    echo ""
    echo "for task in TP53_mutation PIK3CA_mutation Immune_class; do"
    echo "  python src/train_abmil.py \\"
    echo "    --foundational_model ctranspath \\"
    echo "    --latent_dim 768 \\"
    echo "    --work_dir ${WORK_DIR} \\"
    echo "    --train_source cptac_brca \\"
    echo "    --tissue_patching 20x_224px_0px_overlap \\"
    echo "    --task_name \$task \\"
    echo "    --epochs ${EPOCHS} \\"
    echo "    --fold all &"
    echo "done"
    echo "wait"
    echo ""
}

##############################################################################
# EXAMPLE 7: Larger Dataset
##############################################################################
example_large_dataset() {
    echo -e "\n${GREEN}EXAMPLE 7: Large Dataset Example (CPTAC-COAD)${NC}"
    echo "CPTAC-COAD is the largest available dataset (327 samples)"
    echo ""
    echo "python src/train_abmil.py \\"
    echo "    --foundational_model uni_v2 \\"
    echo "    --latent_dim 768 \\"
    echo "    --work_dir ${WORK_DIR} \\"
    echo "    --train_source cptac_coad \\"
    echo "    --tissue_patching 20x_224px_0px_overlap \\"
    echo "    --task_name TP53_mutation \\"
    echo "    --epochs ${EPOCHS} \\"
    echo "    --fold all"
    echo ""
}

##############################################################################
# EXAMPLE 8: Specific Fold Training
##############################################################################
example_specific_fold() {
    echo -e "\n${GREEN}EXAMPLE 8: Training Specific Fold (Testing/Debugging)${NC}"
    echo "Train on a single fold for testing (much faster)"
    echo ""
    echo "python src/train_abmil.py \\"
    echo "    --foundational_model ctranspath \\"
    echo "    --latent_dim 768 \\"
    echo "    --work_dir ${WORK_DIR} \\"
    echo "    --train_source cptac_brca \\"
    echo "    --tissue_patching 20x_224px_0px_overlap \\"
    echo "    --task_name TP53_mutation \\"
    echo "    --epochs 10 \\"
    echo "    --fold 0"  # Only fold 0
    echo ""
}

##############################################################################
# EXAMPLE 9: Available Models Comparison
##############################################################################
example_model_comparison() {
    echo -e "\n${GREEN}EXAMPLE 9: Compare All Available Models${NC}"
    echo "Train all available foundational models (requires lots of compute)"
    echo ""
    echo "models=("
    echo "  'ctranspath:768'"
    echo "  'uni_v1:768'"
    echo "  'uni_v2:768'"
    echo "  'virchow_v1:2560'"
    echo "  'virchow2:2560'"
    echo "  'conch_v1_5:768'"
    echo "  'phikon_v2:768'"
    echo ")"
    echo ""
    echo "for model_spec in \"\${models[@]}\"; do"
    echo "  IFS=':' read model latent_dim <<< \"\$model_spec\""
    echo "  python src/train_abmil.py \\"
    echo "    --foundational_model \$model \\"
    echo "    --latent_dim \$latent_dim \\"
    echo "    --work_dir ${WORK_DIR} \\"
    echo "    --train_source cptac_brca \\"
    echo "    --tissue_patching 20x_224px_0px_overlap \\"
    echo "    --task_name TP53_mutation \\"
    echo "    --epochs ${EPOCHS} \\"
    echo "    --fold all &"
    echo "done"
    echo "wait"
    echo ""
}

##############################################################################
# EXAMPLE 10: Verification
##############################################################################
example_verification() {
    echo -e "\n${GREEN}EXAMPLE 10: Verify PARADIS Setup${NC}"
    echo "Check that all data is properly accessible"
    echo ""
    echo "python verify_paradis_setup.py"
    echo ""
}

##############################################################################
# Main Menu
##############################################################################
show_menu() {
    echo -e "\n${BLUE}Select an example to display:${NC}"
    echo "  1) Single Model Training"
    echo "  2) Multi-Model Training (Sequential)"
    echo "  3) Multi-Model Training (Parallel)"
    echo "  4) Simple Weighted Ensemble"
    echo "  5) Meta-Learner Ensemble"
    echo "  6) Multiple Tasks"
    echo "  7) Large Dataset (CPTAC-COAD)"
    echo "  8) Specific Fold Training"
    echo "  9) All Models Comparison"
    echo "  10) Verification Script"
    echo "  0) Show All Examples"
    echo "  q) Quit"
    echo ""
}

# Main execution
if [ $# -eq 0 ]; then
    # No arguments: show menu
    show_menu
    read -p "Enter choice: " choice

    case $choice in
        1) example_single_model ;;
        2) example_multi_model_seq ;;
        3) example_multi_model_parallel ;;
        4) example_weighted_ensemble ;;
        5) example_meta_learner ;;
        6) example_multiple_tasks ;;
        7) example_large_dataset ;;
        8) example_specific_fold ;;
        9) example_model_comparison ;;
        10) example_verification ;;
        0)
            example_single_model
            example_multi_model_seq
            example_multi_model_parallel
            example_weighted_ensemble
            example_meta_learner
            example_multiple_tasks
            example_large_dataset
            example_specific_fold
            example_model_comparison
            example_verification
            ;;
        q) echo "Goodbye!"; exit 0 ;;
        *) echo "Invalid choice"; exit 1 ;;
    esac
else
    # With argument: run specific example
    case $1 in
        single) example_single_model ;;
        multi_seq) example_multi_model_seq ;;
        multi_parallel) example_multi_model_parallel ;;
        ensemble_weighted) example_weighted_ensemble ;;
        ensemble_meta) example_meta_learner ;;
        tasks) example_multiple_tasks ;;
        large) example_large_dataset ;;
        fold) example_specific_fold ;;
        compare) example_model_comparison ;;
        verify) example_verification ;;
        all)
            example_single_model
            example_multi_model_seq
            example_multi_model_parallel
            example_weighted_ensemble
            example_meta_learner
            example_multiple_tasks
            example_large_dataset
            example_specific_fold
            example_model_comparison
            example_verification
            ;;
        *)
            echo "Usage: $0 [single|multi_seq|multi_parallel|ensemble_weighted|ensemble_meta|tasks|large|fold|compare|verify|all]"
            exit 1
            ;;
    esac
fi

echo -e "\n${BLUE}For more details, see:${NC}"
echo "  • README.md - Full documentation"
echo "  • PARADIS_COMPATIBILITY.md - PARADIS-specific notes"
echo "  • src/ - Implementation scripts"
