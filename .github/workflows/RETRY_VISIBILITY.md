# Understanding Retry Visibility in GitHub Actions

## The Problem

When using the `nick-fields/retry@v3` action, **retries are not visible in the GitHub Actions UI timeline**. Here's why:

### What You See in the UI (Transient Failure - Retry Succeeds)
```
✅ Setup alt e2eamdshark python venv
✅ Run HF top-1000 model
```

### What Actually Happened (Hidden)
```
❌ Setup alt e2eamdshark python venv (Attempt 1) - FAILED
   └─ Retrying...
✅ Setup alt e2eamdshark python venv (Attempt 2) - SUCCESS
✅ Run HF top-1000 model (Attempt 1) - SUCCESS
```

### What You See in the UI (Persistent Failure - All Retries Fail)
```
❌ Setup alt e2eamdshark python venv - FAILED
```

### What Actually Happened
```
❌ Setup alt e2eamdshark python venv (Attempt 1) - FAILED
   └─ Retrying...
❌ Setup alt e2eamdshark python venv (Attempt 2) - FAILED
   └─ Step failed, Job failed ❌
```

**Important:** If all retry attempts fail, the job DOES fail as expected. The retry mechanism only helps with transient failures, not permanent ones.

## Why This Happens

The `nick-fields/retry@v3` action wraps your command and handles retries internally. GitHub Actions only sees the **final result**:
- If any attempt succeeds → Shows ✅ (success) - intermediate failures hidden
- If all attempts fail → Shows ❌ (failure) - **job fails correctly**

The individual retry attempts are logged but **not exposed as separate job steps** in the UI.

### Key Point: Jobs Still Fail Correctly

**The retry mechanism does NOT suppress permanent failures.** If your step fails on all retry attempts:
1. ❌ The step is marked as failed
2. ❌ The job is marked as failed
3. ❌ Subsequent steps are skipped (unless using `continue-on-error`)
4. ❌ The workflow shows as failed

**Retries only help with transient issues** (network glitches, temporary service outages, etc.). Permanent errors like missing files, syntax errors, or configuration issues will still fail the job after all retries are exhausted.

## How to See If Retries Occurred

### Method 1: Look for Warning Annotations (Recommended)

We've added `on_retry_command` to both retry steps:

```yaml
- name: "Setup alt e2eamdshark python venv"
  uses: nick-fields/retry@v3
  with:
    on_retry_command: echo "::warning::Setup venv failed, retrying..."
    # ... other config
```

**Where to find it:**
1. Go to the workflow run
2. Click on the job (e.g., "Models :: rocm :: hf-fill-mask-shard")
3. Expand the step (e.g., "Setup alt e2eamdshark python venv")
4. Look for a yellow warning annotation that says "Setup venv failed, retrying..."

### Method 2: Check Step Logs

Open the step logs and search for:
- Error messages from the first attempt
- The retry action's output showing "Retrying..."
- The second attempt's output

### Method 3: Check Job Duration

If a step took much longer than usual (e.g., 60 minutes instead of 30), it likely retried.

## Example: Detecting a Hidden Retry

### Scenario
Your workflow run shows: `✅ All jobs passed`

But in the step logs you see:
```
ERROR: Failed to install dependencies (connection timeout)
⚠️  Setup venv failed, retrying...
Starting retry attempt 2...
Successfully installed all dependencies
```

This means:
- ❌ First attempt failed (hidden from UI)
- ✅ Second attempt succeeded (shown in UI)
- Result: Step marked as successful, but a retry occurred

## Comparison with Other Retry Approaches

### Option A: Current Approach (nick-fields/retry)
**Pros:**
- Simple configuration
- Automatic retry without code changes
- Works at step level

**Cons:**
- ⚠️ Retries are hidden from UI
- Can only see retries in logs
- Final status doesn't indicate a retry happened

### Option B: Job-level Rerun (Manual)
```yaml
# User manually clicks "Re-run failed jobs" in GitHub UI
```
**Pros:**
- Fully visible in UI
- Clear separation of attempts

**Cons:**
- ❌ Requires manual intervention
- ❌ Slower (re-runs entire job)
- ❌ Wastes CI resources

### Option C: Workflow Retry (Re-run)
```yaml
# If entire workflow fails, GitHub allows re-running all jobs
```
**Pros:**
- Visible as separate workflow runs

**Cons:**
- ❌ Very slow (re-runs everything)
- ❌ Fully manual
- ❌ Expensive in CI time

### Option D: Custom Retry Logic
```yaml
- name: Setup with custom retry
  run: |
    for i in {1..2}; do
      echo "Attempt $i"
      if ./setup.sh; then
        echo "Success!"
        exit 0
      fi
      echo "::warning::Attempt $i failed, retrying..."
      sleep 5
    done
    exit 1
```
**Pros:**
- Full control over logging
- Can make retries very visible

**Cons:**
- More complex to implement
- Need to add to every step that needs retry
- Error handling complexity

## Our Recommendation

**Use the current approach** (nick-fields/retry with `on_retry_command`):

1. ✅ Automatic retry saves manual work
2. ✅ Warning annotations make retries discoverable
3. ✅ Simple configuration
4. ✅ Works well for transient failures

**When to investigate:**
- If you see warning annotations frequently → May indicate a systemic issue
- If steps take unusually long → Check if retries are happening
- If failures persist after retry → Need to fix underlying issue

## Verifying Retry Behavior

To see retry behavior in action:

1. **Monitor actual workflow runs** for retry warnings
2. **Check step logs** when jobs succeed after taking longer than usual
3. **Look for warning annotations** in the GitHub Actions UI

## Key Takeaways

- ✅ Retries ARE happening (they're working correctly)
- ⚠️ Retries are NOT visible in the workflow timeline
- 🔍 Check step logs and warning annotations to see retries
- 📝 This is expected behavior for the retry action, not a bug

## Further Reading

- [nick-fields/retry documentation](https://github.com/nick-fields/retry)
- [GitHub Actions: Workflow commands](https://docs.github.com/en/actions/writing-workflows/choosing-what-your-workflow-does/workflow-commands-for-github-actions)
- [GitHub Actions: Re-running workflows and jobs](https://docs.github.com/en/actions/managing-workflow-runs-and-deployments/managing-workflow-runs/re-running-workflows-and-jobs)
