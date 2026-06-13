CLIP text-tower cache for offline inference.

Layout:
  clip_cache/hub/models--openai--clip-vit-base-patch32/snapshots/<hash>/

On the training machine this may be a symlink to ~/.cache/huggingface/...
On the deployment machine, copy or symlink the full models--openai--clip-vit-base-patch32
directory into clip_cache/hub/.
