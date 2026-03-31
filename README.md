# Mini Agent

这个仓库当前最完整、最有特色的能力是 `VoiceAgent`：一个基于 `Qwen3.5-Omni` 的多模态语音助手，支持：

1. 文本输入
2. 语音输入
3. 模型直接输出文本和语音
4. 在需要时调用工具，再由 `Qwen3.5-Omni` 输出最终语音答复

和传统 `ASR + LLM + TTS` 方案不同，这里走的是更接近端到端的全模态链路：

`用户文本/语音 -> Qwen3.5-Omni -> 工具调用(可选) -> Qwen3.5-Omni 文本+语音 -> 直接播放`

技术架构说明见：

- [VoiceAgent 技术文档](./docs/voice-agent-architecture.md)

## 项目结构

和语音相关的代码已经收拢到 `src/voice/`：

- [src/voice/agent.py](./src/voice/agent.py)：主流程编排
- [src/voice/io.py](./src/voice/io.py)：录音、音频消息构造、流式音频解码、播放器控制
- [src/voice/ui.py](./src/voice/ui.py)：PromptToolkit + Rich 界面
- [src/voice/config.py](./src/voice/config.py)：运行配置
- [src/prompts/voice_prompt.py](./src/prompts/voice_prompt.py)：系统提示词
- [voice_main.py](./voice_main.py)：运行入口

## Python 依赖

本项目使用 `uv` 管理依赖，Python 要求见 [pyproject.toml](./pyproject.toml)：

- Python `>= 3.13`

安装 Python 依赖：

```bash
uv sync
```

如果你还没有安装 `uv`：

```bash
pip install uv
```

## 系统依赖

要运行 `VoiceAgent`，除了 Python 依赖，还需要这些系统工具：

1. `mpv`
   用于播放 `Qwen3.5-Omni` 返回的流式音频
2. `parec` 或 `ffmpeg`
   用于本地录音

### 当前录音后端优先级

`VoiceAgent` 当前的录音逻辑是：

1. 优先尝试 `parec`
2. 如果 `parec` 不可用，则尝试 `ffmpeg`

也就是说，只装其中一个录音工具就可以，但推荐优先保证 `parec` 可用。

## Linux 安装

Linux 是当前 `VoiceAgent` 最适合、也最推荐的运行环境。

### Ubuntu / Debian

常见安装方式：

```bash
sudo apt update
sudo apt install -y mpv ffmpeg pulseaudio-utils
```

说明：

- `pulseaudio-utils` 通常会提供 `parec`
- 如果你的桌面环境使用的是 PipeWire，系统里通常也会带有 PulseAudio 兼容层，因此 `parec` 仍然可用

### Fedora

常见安装方式：

```bash
sudo dnf install -y mpv ffmpeg pulseaudio-utils
```

说明：

- 有些 Fedora 环境需要先启用 RPM Fusion 才能直接安装 `ffmpeg`

### Arch Linux

常见安装方式：

```bash
sudo pacman -S --needed mpv ffmpeg libpulse
```

说明：

- `parec` 通常来自 `libpulse`

## macOS 安装

### Homebrew

先安装 Homebrew，然后安装：

```bash
brew install mpv ffmpeg pulseaudio
```

说明：

1. `ffmpeg` 官方提供源码和第三方编译包信息  
   参考：FFmpeg 官方下载页  
   https://ffmpeg.org/download.html
2. `parec` 来自 `pulseaudio`
3. 当前这套 `VoiceAgent` 录音后端默认基于 `parec` 或 `ffmpeg` 的 PulseAudio 录音输入，因此 macOS 上如果没有 PulseAudio 兼容录音环境，可能需要额外调试

如果你只想先验证“播放和文本对话”链路，`mpv + ffmpeg` 可以先装起来，但要想顺畅录音，通常还需要把本地音频环境配好。

## Windows 安装

### 播放器和 FFmpeg

Windows 可以安装：

1. `mpv`
2. `ffmpeg`

`ffmpeg` 官方下载入口：

- https://ffmpeg.org/download.html

### 录音说明

请注意：当前仓库里的 `VoiceAgent` 录音实现是围绕 `parec` 和 `ffmpeg -f pulse` 写的，因此 **Linux/类 Linux 音频栈是第一优先环境**。

这意味着：

1. 如果你在原生 Windows 下运行，单独安装 `ffmpeg` 往往还不够
2. 你还需要有可用的 PulseAudio 兼容录音输入，或者你需要自行扩展代码支持 Windows 原生录音后端

更现实的建议是：

1. 优先在 Linux 下运行
2. 或者在支持音频设备转发的 WSL / 远程 Linux 环境下运行

## 安装后自检

安装完成后，建议先检查这些命令：

```bash
mpv --version
ffmpeg -version
parec --help
```

如果 `parec` 不存在，也可以只保留：

```bash
mpv --version
ffmpeg -version
```

但此时录音是否能工作，要看 `ffmpeg` 是否带有合适的录音输入后端。

## 配置环境变量

运行前至少需要设置：

```bash
export DASHSCOPE_API_KEY=你的Key
```

常用可选配置：

```bash
export DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
export QWEN_OMNI_MODEL=qwen3.5-omni-plus
export QWEN_OMNI_VOICE=Tina
export QWEN_ENABLE_SEARCH=0
export VOICE_AGENT_MAX_HISTORY_TURNS=3
export VOICE_AGENT_INPUT_SAMPLE_RATE=16000
export VOICE_AGENT_PCM_SAMPLE_RATE=24000
export VOICE_AGENT_PCM_CHANNELS=1
export VOICE_AGENT_PCM_SAMPLE_WIDTH=2
export VOICE_AGENT_TAIL_PADDING_MS=180
export VOICE_AGENT_PLAYBACK_SPEED=1.08
```

这些变量对应的配置定义在：

- [src/voice/config.py](./src/voice/config.py)

## 如何运行 VoiceAgent

入口文件是：

- [voice_main.py](./voice_main.py)

启动命令：

```bash
uv run python voice_main.py
```

启动后支持两种交互方式：

1. 直接输入文本
   例如：
   ```text
   你好
   ```
2. 直接按回车开始录音，再按一次回车结束录音

退出方式：

```text
exit
```

## VoiceAgent 的工作流程

一次完整交互通常会经历这些步骤：

1. 用户输入文本，或者录一段音
2. 如果是语音，则本地录音后转成 `input_audio`
3. 把消息发给 `Qwen3.5-Omni`
4. 模型流式返回文本与音频
5. 如果模型需要工具，则先触发 function calling
6. 工具执行完毕后，再把工具结果回填给模型
7. 模型生成最终答复与最终语音
8. 本地 `mpv` 直接播放返回的流式音频

## 当前实现的几个关键点

### 1. 不是传统 ASR + LLM + TTS

这里不是：

`录音 -> ASR 转文字 -> LLM -> TTS`

而是：

`录音 -> Qwen3.5-Omni -> 直接返回语音`

### 2. 工具调用轮次不会播放中间音频

当模型一轮响应已经进入 `tool_calls` 阶段时，当前轮音频会被停止并丢弃，避免用户听到工具调用中间轮的半成品音频或噪声。

### 3. 播放前会做 PCM 帧对齐

为了减少尾部噪音，当前播放链路会在写给 `mpv` 前先做 PCM 帧对齐，只把完整帧写出去。

### 4. 播放器收尾带静音 padding

播放器关闭前会补一个很短的静音尾巴，避免最后一个字被切掉。

## 常见问题

### 1. `recording failed: Unknown input format: 'pulse'`

说明你的 `ffmpeg` 没有可用的 PulseAudio 录音输入。  
优先安装并使用 `parec`，通常更稳。

### 2. 调工具时会有杂音

当前实现已经做了工具调用轮静音处理。  
如果你还听到异常声音，优先确认是不是旧版本代码，或者工具调用后的最终答复轮本身音频存在问题。

### 3. 尾音有噪声或像没播完

当前代码已经做了几层修复：

1. 播放结束不再 5 秒强杀播放器
2. PCM 帧对齐
3. 尾部静音 padding

如果你依然听到异常，建议下一步把最终 PCM/WAV 落盘，对比“实时播放”和“文件回放”是否一致。

### 4. 为什么文本输入有时没有工具调用

工具调用依赖模型的 function calling 决策。  
当前已经通过系统提示词和兼容逻辑尽量增强稳定性，但 OpenAI-compatible 接口下的 function calling 行为仍然受模型实际返回影响。

## 运行测试

```bash
uv run pytest tests/test_voice_agent.py
```

当前测试覆盖：

1. base64 分片解码
2. PCM 不完整尾帧丢弃
3. 音频输入消息构造
4. 多轮历史裁剪
5. tool call 参数分片拼接
6. 一轮流式响应中的文本、音频、usage 处理
7. tool call 触发后继续下一轮生成最终答复
8. tool call 轮次出现时停止当前音频播放

## 提交前建议

如果你准备提交到 Git，建议至少做一次：

```bash
python -m py_compile src/voice/*.py voice_main.py
uv run pytest tests/test_voice_agent.py
```

这样可以快速确认语法和核心链路没有回归。
