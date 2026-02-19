# Testing Retry Behavior: Success vs Failure

## Quick Reference

| Scenario | Attempt 1 | Attempt 2 | Step Result | Job Result |
|----------|-----------|-----------|-------------|------------|
| **Transient failure** | ❌ Fail | ✅ Pass | ✅ Success | ✅ Success |
| **Persistent failure** | ❌ Fail | ❌ Fail | ❌ Failed | ❌ Failed |
| **Immediate success** | ✅ Pass | (not run) | ✅ Success | ✅ Success |

## Verification

### Current Configuration in test_e2e_hf_top_1000.yml

```yaml
- name: "Setup alt e2eamdshark python venv"
  uses: nick-fields/retry@v3
  with:
    timeout_minutes: 30
    max_attempts: 2      # ← Will try twice
    retry_on: error      # ← Retry on non-zero exit code
    on_retry_command: echo "::warning::Setup venv failed, retrying..."
```

**No `continue-on-error: true`** → Job will fail if all attempts fail ✅

### Expected Behaviors

#### Scenario 1: Transient Failure (Retry Succeeds)
```
Step: Setup alt e2eamdshark python venv
├─ Attempt 1: pip install fails (network timeout)
│  └─ Exit code: 1 ❌
├─ on_retry_command: "::warning::Setup venv failed, retrying..."
└─ Attempt 2: pip install succeeds
   └─ Exit code: 0 ✅

Result: Step ✅ | Job ✅ | Workflow ✅
```

**What you see in GitHub UI:**
- ✅ Green checkmark on step
- ⚠️ Warning annotation in logs (if you expand the step)

#### Scenario 2: Persistent Failure (All Retries Fail)
```
Step: Setup alt e2eamdshark python venv
├─ Attempt 1: pip install fails (package not found)
│  └─ Exit code: 1 ❌
├─ on_retry_command: "::warning::Setup venv failed, retrying..."
└─ Attempt 2: pip install fails again (package still not found)
   └─ Exit code: 1 ❌

Result: Step ❌ | Job ❌ | Workflow ❌
```

**What you see in GitHub UI:**
- ❌ Red X on step
- ❌ Red X on job
- ❌ Red X on workflow run
- Subsequent steps are SKIPPED

#### Scenario 3: Immediate Success (No Retry Needed)
```
Step: Setup alt e2eamdshark python venv
└─ Attempt 1: pip install succeeds
   └─ Exit code: 0 ✅

Result: Step ✅ | Job ✅ | Workflow ✅
(Attempt 2 never runs)
```

**What you see in GitHub UI:**
- ✅ Green checkmark on step
- No warning annotations

## Testing This Locally

You can test the retry behavior using the demo workflow:

```bash
# Trigger the demo workflow
gh workflow run example_retry_demo.yml

# Watch it run
gh run watch
```

The demo includes three test cases:
1. **demo-without-retry**: May fail randomly (50% chance)
2. **demo-with-retry**: Fails on attempt 1, succeeds on attempt 2
3. **demo-all-retries-fail**: Fails on BOTH attempts → Job fails ❌

### Expected Results

After running the demo:
- ❌ `demo-without-retry` - May fail (no retry to save it)
- ✅ `demo-with-retry` - Succeeds after retry
- ❌ `demo-all-retries-fail` - **Fails even with retry** (demonstrates job failure)
- ✅ `realistic-example` - Succeeds after retry
- ✅ `comparison-summary` - Shows summary (runs even if others failed)

## Confirming Job Failure on Exhausted Retries

### Manual Test

You can manually verify the failure behavior by creating a test step:

```yaml
- name: Test permanent failure with retry
  uses: nick-fields/retry@v3
  with:
    timeout_minutes: 2
    max_attempts: 2
    retry_on: error
    on_retry_command: echo "::warning::Retrying..."
    command: |
      echo "This will always fail"
      exit 1  # Always exits with error
```

**Expected result:**
1. Attempt 1 fails with exit code 1
2. Retry warning logged
3. Attempt 2 fails with exit code 1
4. **Step fails** ❌
5. **Job fails** ❌
6. **Workflow fails** ❌

### Real World Example

If your HF model run step fails twice due to a real issue:

```yaml
- name: Run HF top-1000 model
  uses: nick-fields/retry@v3
  with:
    max_attempts: 2
    on_retry_command: echo "::warning::HF model run failed, retrying..."
    command: |
      python3.11 ./run.py ...
      # If this exits with code 1 twice, the job WILL fail
```

**Possible permanent failures that will fail the job:**
- Python syntax error in run.py
- Missing required files
- Out of memory (not transient)
- Invalid command line arguments
- Missing environment variables

**Transient failures that retry might fix:**
- Network timeout downloading model
- Temporary disk space issue (if cleanup happens)
- Race condition in file access
- Temporary GPU driver glitch

## Validation Checklist

✅ Retries work for transient failures (step succeeds on retry)
✅ Warnings are logged when retries occur
✅ Jobs fail when all retry attempts fail
✅ No silent failures - permanent errors still fail the job
✅ Retry attempts are visible in step logs

## Common Misconceptions

❌ **WRONG**: "Retry means my job will never fail"
✅ **CORRECT**: "Retry means my job won't fail due to *transient* issues"

❌ **WRONG**: "If retry is enabled, failures are hidden"
✅ **CORRECT**: "Retry successes hide initial failures, but retry failures still fail the job"

❌ **WRONG**: "I can't see if retries happened"
✅ **CORRECT**: "Retries are visible via ::warning:: annotations in step logs"

## Summary

The retry configuration in your workflow:
1. ✅ **Will retry** on transient failures
2. ✅ **Will fail the job** if all attempts fail
3. ✅ **Will log warnings** when retries occur
4. ✅ **Will not hide** permanent failures

**Your jobs will still fail when they should fail.** The retry mechanism only provides a second chance for transient issues, not a way to hide real problems.
