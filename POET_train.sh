# Training script, will automatically run evaluation also. 
for var in "$@"
do
  echo "Few shot file set $var"
  echo "without eta in inc steps"

  echo "Prompt_642564, training CTR-GCN prompt offset tuning, step 1"
  python poet_main.py --config config/nturgbd-cross-subject/temp_24nov.yaml --device 3 --labels_prev_step 40 --maxlabelid_curr_step 45 --k_shot 5 --weights 'work_dir/ntu60/csub/ctrgcn_prompt/Prompt_642564step0_add_cosinecls_wsig-50.pt' --query_checkpoint 'work_dir/ntu60/csub/ctrgcn_prompt/Prompt_642564step0_add_cosinecls_wsig-query-50.pt' --IL_step 1 --save_name_args 'Prompt_POET_multiple_runs_step1' --prompt_sim_reg  --classifier_average_init --class_order 0 --save_numbers_to_csv --few_shot_data_file "/data/prachi_data/few_shot_files/NTU60_5shots_set${var}.npz" --experiment_name 'POET_ours_16thjuly' --train_eval

  echo "Prompt_642564, training CTR-GCN prompt offset tuning, step 2"
  python poet_main.py --config config/nturgbd-cross-subject/temp_24nov.yaml --device 3 --labels_prev_step 45 --maxlabelid_curr_step 50 --k_shot 5 --weights 'work_dir/ntu60/csub/ctrgcn_prompt/Prompt_POET_multiple_runs_step1-5.pt' --query_checkpoint 'work_dir/ntu60/csub/ctrgcn_prompt/Prompt_POET_multiple_runs_step1-query-5.pt' --IL_step 2 --save_name_args 'Prompt_POET_multiple_runs_step2' --prompt_sim_reg  --classifier_average_init --class_order 0 --save_numbers_to_csv --few_shot_data_file "/data/prachi_data/few_shot_files/NTU60_5shots_set${var}.npz" --experiment_name 'POET_ours_16thjuly' --train_eval

  echo "Prompt_642564, training CTR-GCN prompt offset tuning, step 3"
  python poet_main.py --config config/nturgbd-cross-subject/temp_24nov.yaml --device 3 --labels_prev_step 50 --maxlabelid_curr_step 55 --k_shot 5 --weights 'work_dir/ntu60/csub/ctrgcn_prompt/Prompt_POET_multiple_runs_step2-5.pt' --query_checkpoint 'work_dir/ntu60/csub/ctrgcn_prompt/Prompt_POET_multiple_runs_step2-query-5.pt' --IL_step 3 --save_name_args 'Prompt_POET_multiple_runs_step3' --prompt_sim_reg  --classifier_average_init --class_order 0 --save_numbers_to_csv --few_shot_data_file "/data/prachi_data/few_shot_files/NTU60_5shots_set${var}.npz" --experiment_name 'POET_ours_16thjuly' --train_eval

  echo "Prompt_642564, training CTR-GCN prompt offset tuning, step 4"
  python poet_main.py --config config/nturgbd-cross-subject/temp_24nov.yaml --device 3  --labels_prev_step 55 --maxlabelid_curr_step 60 --k_shot 5 --weights 'work_dir/ntu60/csub/ctrgcn_prompt/Prompt_POET_multiple_runs_step3-5.pt' --query_checkpoint 'work_dir/ntu60/csub/ctrgcn_prompt/Prompt_POET_multiple_runs_step3-query-5.pt' --IL_step 4 --save_name_args 'Prompt_POET_multiple_runs_step4' --prompt_sim_reg --classifier_average_init --class_order 0 --save_numbers_to_csv --few_shot_data_file "/data/prachi_data/few_shot_files/NTU60_5shots_set${var}.npz" --experiment_name 'POET_ours_16thjuly' --train_eval
done