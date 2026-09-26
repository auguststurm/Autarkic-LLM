# Forge Trinity — RTX Pro 2000 16 GB + two RTX Pro 6000 Max-Q

> ⚠️ **Not yet tested** on this three-GPU box (written **2026-09-25**). Confirm load, then first decode, on each GPU before relying on it.

| Server               | Card                                                 |     Port | Weights                                                                | Context | KV          | Sampling                                                      | Output |
| -------------------- | ---------------------------------------------------- | -------: | ---------------------------------------------------------------------- | ------: | ----------- | ------------------------------------------------------------- | -----: |
| **LFM2.5-2.6B**      | RTX Pro 2000 Blackwell **16 GB** (display)           | **8080** | `LFM2.5-2.6B-Q8_0.gguf` (2.87 GB)                                      |  131072 | q8_0 / q8_0 | temp **0.1** · top_k **50** · top_p **1.0** · repeat **1.1**  |  16384 |
| **Qwen3.8-27B**      | RTX Pro 6000 Blackwell Max-Q **96 GB**               | **8081** | `Qwen3.8-27B-UD-Q8_K_XL.gguf` (~31.5 GB)                               |  262144 | q8_0 / q8_0 | temp **0.6** · top_p **0.95** · top_k **20** · min_p **0**    |  16384 |
| **Qwen3-Coder-Next** | the remaining RTX Pro 6000 Blackwell Max-Q **96 GB** | **8082** | `Qwen3-Coder-Next-UD-Q6_K_XL-00001-of-00003.gguf` (~73.1 GB, 3 shards) |  262144 | q8_0 / q8_0 | temp **1.0** · top_p **0.95** · top_k **40** · min_p **0.01** |  32768 |

## Weights

```bash
hf download LiquidAI/LFM2.5-2.6B-GGUF \
  LFM2.5-2.6B-Q8_0.gguf \
  --local-dir ~/Documents/AIML/models

hf download unsloth/Qwen3.8-27B-GGUF \
  Qwen3.8-27B-UD-Q8_K_XL.gguf \
  --local-dir ~/Documents/AIML/models

hf download unsloth/Qwen3-Coder-Next-GGUF \
  --include "UD-Q6_K_XL/*" \
  --local-dir ~/Documents/AIML/models
```

Coder-Next is sharded. Pass `00001-of-00003`. Leave `00002` and `00003` in that folder.

```bash
ls -lh ~/Documents/AIML/models/LFM2.5-2.6B-Q8_0.gguf \
       ~/Documents/AIML/models/Qwen3.8-27B-UD-Q8_K_XL.gguf \
       ~/Documents/AIML/models/UD-Q6_K_XL/
```

## Binary

```bash
cd ~/Documents/GitHub/llama-cpp-turboquant/build/bin
./llama-server --version
./llama-server --list-devices
```

Expect three CUDA devices: one 16 GB (RTX Pro 2000) and two ~96 GB (Max-Q). Qwen3.8 needs `qwen35` in this binary. Coder-Next needs `qwen3next`.

```bash
grep -R LLM_ARCH_QWEN35 ../../src ../../include 2>/dev/null | head -1
grep -R LLM_ARCH_QWEN3NEXT ../../src ../../include 2>/dev/null | head -1
```

## Name the three cards once

After the monitors are unplugged, use the UUID. The CUDA index can change. The two 6000s share a name. The 16 GB row (`memory.total`) is `GPU_LFM`. Pick one 96 GB UUID for `GPU_QWEN38`. The other 96 GB UUID is `GPU_CODER`. Use `pci.bus_id` to tell the 6000s apart.

```bash
nvidia-smi --query-gpu=index,pci.bus_id,name,uuid,memory.total,memory.used --format=csv
```

```bash
cat >> ~/.profile << 'EOF'
# Forge Trinity. UUIDs from nvidia-smi.
# 1. RTX Pro 2000 16 GB — LFM2.5-2.6B :8080
export GPU_LFM=GPU-00000000-0000-0000-0000-000000000000
# 2. RTX Pro 6000 96 GB — Qwen3.8-27B :8081
export GPU_QWEN38=GPU-00000000-0000-0000-0000-000000000000
# 3. RTX Pro 6000 96 GB — Qwen3-Coder-Next :8082
export GPU_CODER=GPU-00000000-0000-0000-0000-000000000000
EOF
source ~/.profile
```

Replace the zeros with the `GPU-<uuid>` values. Each command below should list one device.

```bash
echo "1  RTX 2000  LFM     $GPU_LFM"
echo "2  RTX 6000  Qwen3.8 $GPU_QWEN38"
echo "3  RTX 6000  Coder   $GPU_CODER"
CUDA_VISIBLE_DEVICES="$GPU_LFM" ~/Documents/GitHub/llama-cpp-turboquant/build/bin/llama-server --list-devices
CUDA_VISIBLE_DEVICES="$GPU_QWEN38" ~/Documents/GitHub/llama-cpp-turboquant/build/bin/llama-server --list-devices
CUDA_VISIBLE_DEVICES="$GPU_CODER" ~/Documents/GitHub/llama-cpp-turboquant/build/bin/llama-server --list-devices
```

## Which client may connect

`--host 0.0.0.0`. `llama-server` ignores Pi’s `apiKey`. Each rule names one source address and one of ports 8080, 8081, and 8082.

`192.168.4.43` is this workstation (`ip -4 -br addr`). The address in the rules is the other computer (`hostname -I` there, or the Eero client list). `192.0.2.40` is a stand-in.

Install OpenSSH and allow it, then enable ufw. Keep this desktop login until a second SSH session stays up.

```bash
sudo apt install openssh-server
sudo systemctl enable --now ssh
systemctl is-active ssh
hostname -I
ip -4 -br addr
sudo ufw allow OpenSSH
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow from 192.0.2.40 to any port 8080 proto tcp comment 'lfm'
sudo ufw allow from 192.0.2.40 to any port 8081 proto tcp comment 'qwen3.8'
sudo ufw allow from 192.0.2.40 to any port 8082 proto tcp comment 'coder-next'
sudo ufw enable
sudo ufw status verbose
```

Repeat the three `allow from` lines for each extra computer.

Remove a computer:

```bash
sudo ufw delete allow from 192.0.2.40 to any port 8080 proto tcp
sudo ufw delete allow from 192.0.2.40 to any port 8081 proto tcp
sudo ufw delete allow from 192.0.2.40 to any port 8082 proto tcp
sudo ufw status verbose
```

## Desktop

The RTX Pro 2000 keeps the two monitors. **Ctrl+Alt+F3** is a text login on this same desktop session. The three `llama-server` commands below run here the same way.

`pkill -9 llama-server` stops every server. To stop one, use its own terminal.

## Headless

Boot default stays `graphical.target`. `sudo systemctl isolate multi-user.target` stops GDM and leaves `sshd` up.

### 1. SSH in while the desktop is up

```bash
ssh USER@192.168.4.43
```

Confirm this session before changing the GUI. `systemctl get-default` should print `graphical.target`.

### 2. Leave the GUI

```bash
sudo systemctl isolate multi-user.target
```

The SSH session stays. On a local tty, `exit` returns to the login prompt and leaves the machine on `multi-user.target`.

### 3. Unplug the monitors

Unplug them from the RTX Pro 2000 after GDM has stopped.

```bash
nvidia-smi --query-gpu=index,name,uuid,memory.used,memory.total --format=csv
```

Match the UUIDs to `$GPU_LFM`, `$GPU_QWEN38`, and `$GPU_CODER`. On the 16 GB card, `memory.used` should sit under about 1.5 GB.

If a card disappeared:

```bash
sudo nvidia-smi -pm 1
nvidia-smi -L
```

### 4. Start the servers

One tmux session each, in order: LFM, Qwen3.8, Coder-Next. Run that server’s launch command inside the session. Wait until the port answers, then detach with **Ctrl-b** then **d**. `sudo apt install tmux` if needed.

```bash
tmux new -s lfm
```

```bash
tmux new -s qwen38
```

```bash
tmux new -s coder
```

After each load, `nvidia-smi` should show the gain on that card alone. Reattach with `tmux attach -t lfm` (`qwen38`, `coder`).

### 5. Log out of SSH

Detach first, then `exit`. The servers keep running. `ssh USER@192.168.4.43` logs in again. A server started in the SSH foreground, outside tmux, exits when that session ends.

### 6. Return to the desktop

Stop LFM so the 2000 is free for the displays. Qwen3.8 and Coder-Next can stay up.

```bash
tmux attach -t lfm
# Ctrl-C
```

Plug the monitors into the RTX Pro 2000, then:

```bash
sudo systemctl isolate graphical.target
exit
```

To stop all three first:

```bash
tmux attach -t lfm      # Ctrl-C
tmux attach -t qwen38   # Ctrl-C
tmux attach -t coder    # Ctrl-C
```

`pkill -9 llama-server` stops all three. Then isolate `graphical.target`.

## LFM2.5-2.6B — RTX Pro 2000 (`:8080`)

The chat template always opens `<think>`. This command has no `--reasoning` flag and no `--cache-ram`. `--top-p 1.0` matches Liquid’s generation config: temperature 0.1, top_k 50, repetition penalty 1.1. If prefill OOMs on the desktop, set both batch sizes to **256**, then `--ctx-size 65536` and Pi `contextWindow` to match.

```bash
cd ~/Documents/GitHub/llama-cpp-turboquant/build/bin

CUDA_VISIBLE_DEVICES="$GPU_LFM" ./llama-server \
  --model ~/Documents/AIML/models/LFM2.5-2.6B-Q8_0.gguf \
  --alias lfm2.5-2.6b \
  --host 0.0.0.0 --port 8080 \
  --ctx-size 131072 \
  --fit off \
  --n-gpu-layers 99 \
  --split-mode none \
  --main-gpu 0 \
  --load-mode none \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --jinja \
  --flash-attn on \
  --no-context-shift \
  --parallel 1 \
  --ubatch-size 1024 \
  --batch-size 1024 \
  --repeat-penalty 1.1 \
  --presence-penalty 0.0 \
  --frequency-penalty 0.0 \
  --min-p 0.0 \
  --repeat-last-n 512 \
  --threads 8 --temp 0.1 --top-k 50 --top-p 1.0 \
  --n-predict 16384 \
  --kv-unified \
  --log-verbosity 1
```

## Qwen3.8-27B — RTX Pro 6000 (`:8081`)

```bash
cd ~/Documents/GitHub/llama-cpp-turboquant/build/bin

CUDA_VISIBLE_DEVICES="$GPU_QWEN38" ./llama-server \
  --model ~/Documents/AIML/models/Qwen3.8-27B-UD-Q8_K_XL.gguf \
  --alias qwen3.8-27b \
  --host 0.0.0.0 --port 8081 \
  --ctx-size 262144 \
  --fit off \
  --n-gpu-layers 99 \
  --split-mode none \
  --main-gpu 0 \
  --load-mode none \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --cache-ram 0 \
  --jinja \
  --flash-attn on \
  --no-context-shift \
  --parallel 1 \
  --ubatch-size 1024 \
  --batch-size 1024 \
  --reasoning off \
  --reasoning-budget 0 \
  --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.0 \
  --presence-penalty 0.0 \
  --repeat-penalty 1.0 \
  --frequency-penalty 0.0 \
  --repeat-last-n 64 \
  --threads 12 \
  --n-predict 16384 \
  --kv-unified \
  --log-verbosity 1
```

## Qwen3-Coder-Next — RTX Pro 6000 (`:8082`)

`--load-mode none` reads about 73 GB through host RAM. Start this server after Qwen3.8 is listening.

```bash
cd ~/Documents/GitHub/llama-cpp-turboquant/build/bin

CUDA_VISIBLE_DEVICES="$GPU_CODER" ./llama-server \
  --model ~/Documents/AIML/models/UD-Q6_K_XL/Qwen3-Coder-Next-UD-Q6_K_XL-00001-of-00003.gguf \
  --alias qwen3-coder-next \
  --host 0.0.0.0 --port 8082 \
  --ctx-size 262144 \
  --fit off \
  --n-gpu-layers 99 \
  --split-mode none \
  --main-gpu 0 \
  --load-mode none \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --cache-ram 0 \
  --jinja \
  --flash-attn on \
  --no-context-shift \
  --parallel 1 \
  --ubatch-size 1024 \
  --batch-size 1024 \
  --reasoning off \
  --reasoning-budget 0 \
  --temp 1.0 --top-p 0.95 --top-k 40 --min-p 0.01 \
  --presence-penalty 0.0 \
  --repeat-penalty 1.0 \
  --frequency-penalty 0.0 \
  --repeat-last-n 64 \
  --threads 12 \
  --n-predict 32768 \
  --kv-unified \
  --log-verbosity 1
```

## Confirm

```text
# :8080  LFM         RTX 2000   n_ctx_seq (131072)   kv_cache_init ≈ 1088 MiB
# :8081  Qwen3.8     RTX 6000   n_ctx_seq (262144)   arch qwen35
# :8082  Coder-Next  RTX 6000   n_ctx_seq (262144)   arch qwen3next
```

If a server lists more than one CUDA device, its `GPU_*` variable is empty or wrong. Stop that server and fix the UUID.

These curls run on the workstation. LFM’s trace is in `reasoning_content` and the answer is in `content`. From an allowed client, use `192.168.4.43` in place of `127.0.0.1`.

```bash
curl -s --noproxy '*' http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"lfm2.5-2.6b","messages":[{"role":"user","content":"What is 17 * 23? Reply with just the number."}]}'

curl -s --noproxy '*' http://127.0.0.1:8081/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"What is 17 * 23? Reply with just the number."}]}'

curl -s --noproxy '*' http://127.0.0.1:8082/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3-coder-next","messages":[{"role":"user","content":"What is 17 * 23? Reply with just the number."}]}'
```

`ss -tlnp | grep -E '8080|8081|8082'` on the workstation should show `0.0.0.0` on each port.

## Pi on another computer

```bash
curl -fsSL https://pi.dev/install.sh | sh
pi --version
```

Save the JSON below as `~/.pi/agent/models.json`. Quit Pi and start it again after a change. `contextWindow` matches `--ctx-size`. `maxTokens` matches `--n-predict`. `--alias` matches `id`. Pi hides a provider with an empty `apiKey`, so the value stays `1337`.

LFM uses `reasoning` false for tool calls. The server template still opens `<think>`.

```json
{
  "providers": {
    "llama-cpp-8080": {
      "baseUrl": "http://192.168.4.43:8080/v1",
      "api": "openai-completions",
      "apiKey": "1337",
      "compat": {
        "supportsDeveloperRole": false,
        "supportsReasoningEffort": false
      },
      "models": [
        {
          "id": "lfm2.5-2.6b",
          "name": "LFM2.5-2.6B Q8_0 (128k) - RTX Pro 2000",
          "reasoning": false,
          "contextWindow": 131072,
          "maxTokens": 16384
        }
      ]
    },
    "llama-cpp-8081": {
      "baseUrl": "http://192.168.4.43:8081/v1",
      "api": "openai-completions",
      "apiKey": "1337",
      "compat": {
        "supportsDeveloperRole": false,
        "supportsReasoningEffort": false
      },
      "models": [
        {
          "id": "qwen3.8-27b",
          "name": "Qwen3.8-27B Q8_K_XL (262k) - RTX Pro 6000",
          "reasoning": false,
          "contextWindow": 262144,
          "maxTokens": 16384
        }
      ]
    },
    "llama-cpp-8082": {
      "baseUrl": "http://192.168.4.43:8082/v1",
      "api": "openai-completions",
      "apiKey": "1337",
      "compat": {
        "supportsDeveloperRole": false,
        "supportsReasoningEffort": false
      },
      "models": [
        {
          "id": "qwen3-coder-next",
          "name": "Qwen3-Coder-Next Q6_K_XL (262k) - RTX Pro 6000",
          "reasoning": false,
          "contextWindow": 262144,
          "maxTokens": 32768
        }
      ]
    }
  }
}
```

```text
/model llama-cpp-8080/lfm2.5-2.6b
/model llama-cpp-8081/qwen3.8-27b
/model llama-cpp-8082/qwen3-coder-next
```

Copy this file to each additional computer allowed through ufw.

**Last Updated:** 2026-09-25 (⚠️ untested)
