# Prevent System Sleep During Long-Running Tasks

## Quick Solution: Use `caffeinate`

### Option 1: Prevent Sleep for Current Command (Recommended)

Run your evaluation command with `caffeinate`:

```bash
# For mBERT evaluation
caffeinate -i python scripts/evaluate.py \
    --model mbert \
    --checkpoint models/mbert/best_model.pt \
    --data-path data/squad/dev-v2.0.json \
    --dataset-type squad

# For mT5 evaluation
caffeinate -i python scripts/evaluate.py \
    --model mt5 \
    --checkpoint models/checkpoints/checkpoint_epoch_1.pt \
    --data-path data/squad/dev-v2.0.json \
    --dataset-type squad
```

**What `-i` does**: Prevents the system from idle sleeping (but allows manual sleep via menu)

### Option 2: Prevent Sleep Indefinitely (Until You Cancel)

If you want to prevent sleep for multiple commands:

```bash
# Start caffeinate in background (prevents sleep)
caffeinate -i &

# Note the process ID (PID) shown, then run your command normally
python scripts/evaluate.py --model mbert ...

# When done, kill caffeinate
killall caffeinate
```

### Option 3: Prevent Sleep for Specific Duration

```bash
# Prevent sleep for 2 hours (7200 seconds)
caffeinate -i -t 7200 python scripts/evaluate.py ...
```

## Alternative: System Settings (Manual)

1. **System Preferences** → **Energy Saver** (or **Battery** on newer macOS)
2. Set "Prevent computer from sleeping automatically when the display is off" to **Never**
3. **Note**: This affects all usage, not just your terminal

## Recommended Approach

**For your current situation**, if the evaluation is already running:

1. **Open a new terminal window**
2. **Run this command** to prevent sleep:
   ```bash
   caffeinate -i
   ```
3. **Leave that terminal open** - it will prevent sleep until you close it or press Ctrl+C
4. **Your evaluation will continue** in the other terminal

**Or** if you want to restart the evaluation with sleep prevention:

```bash
# Cancel current evaluation (Ctrl+C)
# Then run with caffeinate:
caffeinate -i python scripts/evaluate.py \
    --model mbert \
    --checkpoint models/mbert/best_model.pt \
    --data-path data/squad/dev-v2.0.json \
    --dataset-type squad
```

## Check if Caffeinate is Running

```bash
ps aux | grep caffeinate | grep -v grep
```

If you see a process, sleep prevention is active.

## Stop Caffeinate

```bash
killall caffeinate
```

---

**Quick Tip**: The `-i` flag prevents idle sleep but still allows:
- Display to sleep (saves power)
- Manual sleep via Apple menu
- System to wake normally

This is the safest option that won't drain your battery unnecessarily.

