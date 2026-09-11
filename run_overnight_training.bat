@echo off
REM Unattended overnight fine-tune + benchmark. Safe to close the Claude session.
REM Logs to logs\overnight.log ; leaves the demo build untouched on failure.
.\.venv-train\Scripts\python.exe tools\run_overnight.py ^
  --manifest "C:/Users/pradz/AppData/Local/Temp/claude/C--Users-pradz-Downloads-web-dev-shi-depth-wizard3D-main-depth-wizard3D-main/2dbf4613-aea0-44de-a25c-2bbaa54a175f/scratchpad\gamus_train_manifest.json" ^
  --test-manifest "C:/Users/pradz/AppData/Local/Temp/claude/C--Users-pradz-Downloads-web-dev-shi-depth-wizard3D-main-depth-wizard3D-main/2dbf4613-aea0-44de-a25c-2bbaa54a175f/scratchpad\gamus_test_manifest.json" ^
  --min-tiles 150 --wait-min 180 --epochs 30
pause
