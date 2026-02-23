# GitHub Actions Step Retry Setup

## Summary

Added automatic retry logic to handle transient failures in the HF Top-1000 models test suite workflow.

## What Was Changed

### Modified Workflows

Retry logic has been added to all network-dependent steps across all workflows. Key examples:

#### 1. Setup Python Virtual Environment (Lines 201-220)
```yaml
- name: "Setup alt e2eamdshark python venv"
  uses: nick-fields/retry@v3
  with:
    timeout_minutes: 30
    max_attempts: 2
    retry_on: error
```

**Why?** This step failed in run #22133201851 due to transient dependency installation issues.

#### 2. Run HF Model Tests (Lines 222-247)
```yaml
- name: Run HF top-1000 model
  uses: nick-fields/retry@v3
  with:
    timeout_minutes: 120
    max_attempts: 2
    retry_on: error
```

**Why?** Prevents test flakiness from causing complete job failures.

## How It Works

- **First Attempt**: Runs normally
- **On Failure**: Automatically retries once more
- **On Success**: Proceeds to next step (no retry needed)
- **After 2 Failures**: **Marks the step AND job as FAILED** ❌

### Important: Jobs Still Fail When All Retries Fail

The retry mechanism does NOT hide permanent failures:
- ✅ **Transient failure** (attempt 1 fails, attempt 2 succeeds) → Job passes
- ❌ **Persistent failure** (both attempts fail) → **Job fails** (as it should)

This ensures that real issues are still caught and the job fails appropriately.

## Example Run Behavior

### Before (without retry):
```
✗ Setup venv → FAILED (transient network issue)
  Job terminates ❌
```

### After (with retry):
```
✗ Setup venv (Attempt 1) → FAILED (transient network issue)
  ↓ Automatic retry
✓ Setup venv (Attempt 2) → SUCCESS
  ↓
✓ Run tests → SUCCESS
  Job completes ✅
```

## Testing the Retry Mechanism

To verify retry functionality is working:

1. **Monitor workflow runs** for warning annotations in step logs
2. **Check step duration** - steps that take ~2x normal time likely retried
3. **Review logs** for retry messages like "::warning::Setup venv failed, retrying..."

See [RETRY_BEHAVIOR_TEST.md](./RETRY_BEHAVIOR_TEST.md) for detailed testing scenarios.

## Limitations

- **Not true job-level retry**: Only retries individual steps, not entire jobs
- **Manual re-runs still available**: You can still manually re-run failed workflows from GitHub UI
- **Same timeout limits apply**: The job timeout (6000 minutes) still applies to total job duration
- **Retry visibility**: The retry action shows only the final outcome in the workflow UI
  - If a step succeeds after retry, it appears as a simple success
  - Failed attempts are logged but not shown as separate failures in the UI
  - Look for "::warning::" messages in step logs to see if retries occurred
  - The `on_retry_command` parameter logs warnings when retries happen

## Configuration Options

Adjust retry behavior by modifying these parameters:

```yaml
timeout_minutes: 30      # Max time per attempt
max_attempts: 2          # Total attempts (1 original + 1 retry)
retry_on: error          # Retry on non-zero exit code
```

For more aggressive retry:
```yaml
max_attempts: 3          # Try up to 3 times
timeout_minutes: 60      # Give more time per attempt
```

## When to Use Retry

✅ **Good use cases:**
- Dependency installation (pip, apt, npm)
- Network operations (downloads, API calls)
- Flaky tests with occasional transient failures
- Cloud service interactions

❌ **Don't use retry for:**
- Steps that deterministically fail (code errors)
- Very long-running operations (already have timeout)
- Steps that modify state (git commits, deployments)

## Reference

- GitHub Action: https://github.com/nick-fields/retry
- Failed run that motivated this change: https://github.com/nod-ai/AMD-SHARK-TestSuite/actions/runs/22133201851/job/63978168038
