$ErrorActionPreference = "Stop"

Set-Location (Split-Path -Parent $PSScriptRoot)

if (-not $env:API_BASE) {
  throw "API_BASE is not set."
}
if (-not $env:API_KEY) {
  throw "API_KEY is not set."
}
if (-not $env:MODEL_NAME) {
  $env:MODEL_NAME = "gpt-5.4"
}

python scripts\run_distill.py `
  --input_dir data\normalized `
  --output_dir data\distilled `
  --sources fincuge_instruction `
  --splits train val `
  --max_concurrency 8 `
  --chunk_size 64 `
  --prompt_version fincuge_instruction_hybrid_v1 `
  --overwrite
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

python scripts\run_fincuge_instruction_answer_score.py `
  --inputs data\distilled\fincuge_instruction\train.jsonl data\distilled\fincuge_instruction\val.jsonl `
  --output_dir data\reports\answer_score_full\fincuge_instruction `
  --filtered_output_dir data\distilled_filtered\fincuge_instruction `
  --max_concurrency 32 `
  --model_name $env:MODEL_NAME
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

python scripts\run_reasoning_quality_score.py `
  --inputs data\distilled_filtered\fincuge_instruction\train.keep.jsonl data\distilled_filtered\fincuge_instruction\val.keep.jsonl `
  --output_dir data\reports\reasoning_quality_score_full `
  --filtered_output_dir data\distilled_filtered_round2\fincuge_instruction `
  --max_concurrency 32 `
  --model_name $env:MODEL_NAME
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

New-Item -ItemType Directory -Force train_data\fincuge_instruction | Out-Null
Copy-Item data\distilled_filtered_round2\fincuge_instruction\train.keep.round2.keep.jsonl train_data\fincuge_instruction\train.jsonl -Force
Copy-Item data\distilled_filtered_round2\fincuge_instruction\val.keep.round2.keep.jsonl train_data\fincuge_instruction\val.jsonl -Force

Write-Output "Pipeline completed successfully."
