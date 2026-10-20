# Preserved first boundary anchor

The first targeted continuation successfully completed a default-allocator cold prefill at 114,688 tokens on GPU-2f21e758-c259-8482-8879-4500abdf0653. Its checkpoint was recovered from the executed Colab output and fsynced locally after the browser session ended. The checkpoint contains the worker result and post-exit idle verification. No deliberate attempt was dispatched in this allocation.

SHA-256: `8c799e0d089f3c9b971ad35a17aef7747282f2cf1868a269726b8b422535ab34`.

After Colab assigned a different physical T4, a new prospective registration was required. This success is preserved as separate evidence and is never pooled into another allocation's boundary or the core fit.
