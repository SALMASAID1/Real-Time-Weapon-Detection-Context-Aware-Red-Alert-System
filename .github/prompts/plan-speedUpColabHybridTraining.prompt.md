## Plan: Speed Up Colab Hybrid Training (T4)

Your 25h/50-epoch time is consistent with a compute-heavy Hybrid model (YOLO11m backbone + Swin neck) at 640px plus avoidable overhead in the v7 notebook: full validation every epoch, large checkpoint writes to Google Drive every epoch, and extra per-batch work in target matching/anchor generation.

This plan prioritizes changes that are (a) fastest to apply in Colab, (b) lowest risk, and (c) deliver the biggest wall-clock reduction.

Baseline from your run: at `1.55 it/s` with `2591` train iterations per epoch (batch=16 on ~41k images), Phase-1 epoch time is ~28 minutes. Training-only time for 50 epochs is ~23–24 hours, and Phase-2 (unfreezing the backbone) will usually be slower than Phase-1.

**Steps**
1. Establish a 5-minute baseline so you know the bottleneck (compute vs input pipeline).
   - In the first training epoch, watch GPU utilization and memory (via `nvidia-smi` in Colab). If GPU util is consistently <70%, data loading/CPU aug is the limiter; if it’s 90–100%, the model/loss is the limiter.
   - Capture: seconds/iter for ~50 iterations, and seconds spent on end-of-epoch validation + checkpoint save.

2. Remove the biggest wasted overheads in the v7 notebook (no accuracy impact).
   - Checkpoint destination: write checkpoints to Colab SSD (`/content/...`) during training; sync/copy to Google Drive only every N epochs (or at the end). Writing directly to Drive each epoch is often the slowest part of the loop.
   - Checkpoint frequency: save `last.pt` every 5 epochs (and at the phase transition); keep `best.pt` updates when validation runs.
   - Resume safety: if you switch to SSD checkpoints, add a small startup step that copies `last.pt` from Drive → SSD when resuming, and sync back to Drive every N epochs.

3. Reduce validation cost (near-zero risk).
   - Validate every N epochs (e.g., 5) instead of every epoch; optionally validate every epoch only during the first 2–3 epochs (sanity) and at the end of each phase.
   - If you must track best model precisely, validate on a fixed subset (e.g., 20–25% of val) during training and run full val once at the end.

4. Increase effective throughput on T4 (usually high impact).
   - Increase `batch_size` from 16 to the maximum stable value on your T4 with AMP enabled (commonly 20–32 depending on model + imgsz). Larger batch reduces iterations/epoch.
   - Keep AMP (already enabled). Enable cuDNN autotuning (`torch.backends.cudnn.benchmark=True`) and use non-blocking H2D copies.

5. Tune the dataloader for Colab (impact depends on your CPU cores).
   - If GPU util is low, try `num_workers` in {2, 4, 6}; enable `persistent_workers=True` (only if `num_workers>0`) and set a small `prefetch_factor`.
   - If GPU util is high, keep `num_workers` modest; the run is compute-bound and extra workers won’t help.

6. If you still can’t hit the deadline, apply controlled model simplifications (small accuracy trade-off, big speed gain) while keeping the Hybrid architecture.
   - Reduce Swin depth: set Swin `num_blocks` from 2 → 1.
   - Use a smaller backbone variant: switch from `yolo11m.pt` → `yolo11n.pt` (or the smallest acceptable variant you have).
   - Optional: reduce Swin compute further (embed_dim and/or num_heads), but do this only after the depth/backbone change because it requires more re-tuning.

7. Make the loss/target matching faster (accuracy-neutral, requires small code edits).
   - Cache anchor centers/strides once per device+imgsz instead of recomputing every batch.
   - Remove Python loops in `compute_loss` foreground anchor center/stride gathering by using `torch.nonzero(fg_mask)`.
   - If still needed, vectorize `build_targets()` to avoid per-GT kernel launches (bigger change; do after the quick wins).

8. (Optional) Resolution adjustment (big speed lever but needs consistent plumbing).
   - If you reduce `imgsz`, update it consistently in: the dataloader, SwinNeck initialization, and DetectionHead anchor generation/normalization (currently hard-coded for 640).
   - Recommended safe “Swin-friendly” sizes are those where `imgsz/16` is divisible by the Swin window size.

**Relevant files**
- notebooks/Phase2_Training_v7.ipynb — dataloader (`num_workers=2`), per-epoch validation, and per-epoch Drive checkpointing
- models/hybrid_model.py — Hybrid assembly; SwinNeck hyperparams (embed_dim/window/num_blocks/imgsz)
- models/necks/swin_neck.py — Swin block stack depth and attention window
- models/heads/detection_head.py — hard-coded imgsz=640 anchor generation + loops in `build_targets()` and `compute_loss()`

**Verification**
1. Re-run 1 epoch and confirm:
   - GPU util stays high (ideally >85%)
   - end-of-epoch overhead (val + save) is reduced
2. Compare wall-clock per epoch before/after; extrapolate expected time for 50 epochs.
3. Confirm resume still works (interrupt Colab, restart, resume from checkpoint).
4. Run the notebook’s visual audit step to ensure decode logic still works.

**Decisions**
- Keep Hybrid architecture (YOLO backbone + Swin neck) on Colab T4.
- Prioritize time-to-result over perfect reproducibility; keep “resume” but avoid expensive Drive writes.

**Further Considerations**
1. If you share your typical VRAM usage at batch=16 (from Colab printout), we can pick a more precise target batch size.
2. If GPU util is low even after increasing workers, we should reduce CPU-heavy augmentations (mosaic/mixup) inside Ultralytics’ dataset pipeline.
