# VoiceAgent 技术文档

## 1. 概述

本项目中的 `VoiceAgent` 是一个基于 `Qwen3.5-Omni` 的多模态语音助手。它支持：

1. 用户通过文本输入提问
2. 用户通过语音输入提问
3. 模型直接返回文本与音频
4. 模型在需要时调用工具，再基于工具结果继续生成最终答复

和传统的 `ASR + LLM + TTS` 三段式语音系统不同，这里的核心思路是：

1. 输入侧不先强依赖独立 ASR 服务做转写
2. 输出侧不再把 LLM 文本结果送给独立 TTS 模块合成
3. 而是直接把语音输入交给 `Qwen3.5-Omni`
4. 再直接播放 `Qwen3.5-Omni` 返回的流式音频

也就是说，`VoiceAgent` 的主要交互链路是：

`用户语音/文本 -> Qwen3.5-Omni -> 工具调用(可选) -> Qwen3.5-Omni 文本+语音输出 -> 直接播放`

## 2. 与传统 ASR + LLM + TTS 方案的区别

### 2.1 传统方案

传统语音助手通常拆成三段：

1. `ASR`：把用户语音转成文本
2. `LLM`：基于文本理解意图并生成答复文本
3. `TTS`：把答复文本再转换成语音

链路可以写成：

`用户语音 -> ASR -> 文本 -> LLM -> 文本 -> TTS -> 语音`

### 2.2 本项目 VoiceAgent 方案

本项目的核心链路是：

`用户语音 -> Qwen3.5-Omni -> 工具调用(可选) -> Qwen3.5-Omni 文本+音频 -> 播放`

如果用户输入的是文本，则直接跳过录音阶段：

`用户文本 -> Qwen3.5-Omni -> 工具调用(可选) -> Qwen3.5-Omni 文本+音频 -> 播放`

### 2.3 相比传统方案的优势

1. 链路更短，系统集成更简单  
   传统方案要维护三个核心模型或服务：ASR、LLM、TTS。当前方案把“语音理解 + 文本推理 + 语音输出”集中到一个核心模型里，系统复杂度更低。

2. 模态一致性更好  
   传统方案里，ASR 转写文本可能已经损失一部分语气、停顿、口语表达特征；TTS 又会用另一套模型重新“朗读”文本。当前方案里，输入理解和输出语音都由同一个全模态模型主导，模态衔接通常更自然。

3. 工程边界更清楚  
   在本项目里，除了本地录音和本地播放，推理主体都围绕同一个模型展开，减少了跨服务协议、重试、时延归因和数据格式转换问题。

4. 端到端语音体验更自然  
   传统三段式经常出现“识别风格”和“播报风格”割裂的问题。这里的输出音频直接来自 `Qwen3.5-Omni`，因此更容易形成统一的助手人格和语音风格。

5. 更容易做多模态扩展  
   当前 `VoiceAgent` 虽然主打语音，但其模型本身就是全模态模型。后续若要扩展图片、音视频输入，不需要推翻整体架构。

### 2.4 相比传统方案的代价

1. 对单一模型能力更依赖  
   如果模型某轮没有返回音频，或工具调用结构化输出不稳定，问题会直接影响整条链路。

2. 调试方式不同  
   传统三段式容易分段排查；全模态方案更依赖对流式事件、兼容接口差异和模型行为的整体理解。

3. 可替换性稍弱  
   三段式可以独立更换 ASR 或 TTS；当前方案的语音体验更多绑定在 `Qwen3.5-Omni` 的行为上。

## 3. 当前实现的模块划分

当前 `VoiceAgent` 已拆成四个主要模块：

1. [agent.py](/home/howard/ai-workspace/mini-agent/src/voice/agent.py)  
   负责对话编排、工具调用、多轮消息管理、流式响应处理

2. [io.py](/home/howard/ai-workspace/mini-agent/src/voice/io.py)  
   负责录音、音频消息构造、流式音频解码、PCM 对齐、本地播放器控制

3. [ui.py](/home/howard/ai-workspace/mini-agent/src/voice/ui.py)  
   负责 `PromptToolkit + Rich` 终端交互体验

4. [voice_prompt.py](/home/howard/ai-workspace/mini-agent/src/prompts/voice_prompt.py)  
   负责语音模式的系统提示词

此外，配置被抽到了：

5. [config.py](/home/howard/ai-workspace/mini-agent/src/voice/config.py)  
   负责模型、音色、搜索开关、历史长度、录音采样率、播放速度、尾部 padding 等参数

## 4. 核心技术流程

## 4.1 启动与配置加载

入口文件是 [voice_main.py](/home/howard/ai-workspace/mini-agent/voice_main.py)。

启动时主要执行：

1. 创建 `VoiceAgent`
2. 从环境变量加载 `VoiceAgentConfig`
3. 初始化 `OpenAI-compatible` 客户端
4. 初始化 `VoiceUI`
5. 准备系统消息与多轮历史

关键点：

1. `VoiceAgentConfig.from_env()` 从环境变量统一读取配置
2. `VoiceAgent` 不再直接硬编码播放速度、采样率、模型名等参数
3. 系统提示词由 `build_voice_system_prompt()` 提供

## 4.2 用户输入阶段

在 [agent.py](/home/howard/ai-workspace/mini-agent/src/voice/agent.py) 中，`run()` 主循环支持两种输入：

1. 文本输入
2. 语音输入

逻辑入口是：

1. `PromptToolkit` 读取 `voice> ` 提示符输入
2. 如果用户输入非空文本，则直接追加为：
   `{"role": "user", "content": "<text>"}`
3. 如果用户直接回车，则进入录音逻辑

对应方法：

1. `_append_user_input()`
2. `_wait_for_record_stop()`
3. `record_audio()`
4. `build_audio_message()`

## 4.3 录音实现

录音逻辑在 [io.py](/home/howard/ai-workspace/mini-agent/src/voice/io.py)。

当前实现优先使用：

1. `parec`
2. 如果不可用，则尝试 `ffmpeg`

核心步骤：

1. 生成临时 WAV 文件路径
2. 启动录音子进程
3. 等待用户按回车停止录音
4. 向录音进程发送 `SIGINT`
5. 读取临时音频文件字节
6. 删除临时文件

这里有两个工程上的关键点：

1. 不直接依赖 Python 麦克风库，减少平台兼容问题
2. 使用外部录音工具生成标准音频文件，便于后续直接作为模型输入

## 4.4 语音消息构造

音频输入不是先做 ASR，而是直接传给模型。

实现位于 [io.py](/home/howard/ai-workspace/mini-agent/src/voice/io.py) 的 `build_audio_message()`。

构造后的消息格式是：

```json
{
  "role": "user",
  "content": [
    {
      "type": "input_audio",
      "input_audio": {
        "data": "data:;base64,...",
        "format": "wav"
      }
    }
  ]
}
```

关键逻辑：

1. 音频文件读成 bytes
2. 做 base64 编码
3. 包装成 `data:;base64,...` 形式
4. 按 `OpenAI-compatible` 的 `input_audio` 内容结构提交

这一步就是 VoiceAgent 和传统 ASR 流程的核心差别之一：  
这里没有“先转文本再发给模型”的中间步骤。

## 4.5 模型请求与流式响应

主响应逻辑在 [agent.py](/home/howard/ai-workspace/mini-agent/src/voice/agent.py) 中的：

1. `_stream_response()`
2. `_run_model_turn()`
3. `_consume_delta()`

模型请求参数的关键点：

1. `modalities=["text", "audio"]`
2. `audio={"voice": ..., "format": "wav"}`
3. `stream=True`
4. `tools=self._tool_set.to_schemas()`

也就是说，每一轮模型既可能：

1. 直接返回文本与音频
2. 也可能先返回 `tool_calls`

## 4.6 工具调用逻辑

工具调用仍然遵循项目已有的 function calling 模式。

关键方法：

1. `_collect_tool_calls()`
2. `_append_tool_call_messages()`
3. `_run_tool_calls()`

工作方式：

1. 在流式过程中累计 `delta.tool_calls`
2. 将分片化的函数参数拼接成完整 JSON 参数串
3. 一旦这一轮出现工具调用，就把 assistant 的 `tool_calls` 消息写回历史
4. 执行工具
5. 把工具输出作为 `tool` 消息追加回历史
6. 再次请求模型生成最终自然语言答复与音频

这一点非常关键：  
当前 VoiceAgent 并不是“工具执行后自己拼文本回答”，而是“工具执行后再交回给 `Qwen3.5-Omni` 继续生成最终答复和语音”。

这也是它和很多“LLM 负责文本、TTS 负责声音”的代理式语音系统的重要不同。

另外，当前实现还专门处理了一个语音代理里很常见但容易被忽略的问题：

1. 当一轮响应已经进入 `tool_calls` 阶段时，这一轮的音频并不一定是最终可播报内容
2. 如果继续直接播放这类中间轮音频，可能会出现短促、异常或嘈杂的声音
3. 因此当前实现会在检测到 `tool_calls` 后立即停止当前轮播放器，并丢弃该轮剩余音频
4. 最终只播放工具执行完成后的“最终答复轮”音频

这保证了用户听到的语音始终尽量是稳定、完整、面向用户的最终答复，而不是中间推理或工具轮次的半成品音频。

## 4.7 音频输出与实时播放

音频播放逻辑位于 [io.py](/home/howard/ai-workspace/mini-agent/src/voice/io.py)。

模型返回的是流式音频分片，关键步骤如下：

1. 从 `delta.audio.data` 或 `delta.model_extra["audio"]` 中提取 base64 音频分片
2. 使用 `Base64StreamDecoder` 处理“跨 chunk 分片”的 base64 数据
3. 逐步解码为 PCM 字节流
4. 使用 `PCMFrameBuffer` 对 PCM 数据做帧对齐，只把完整帧写给播放器
5. 将对齐后的 PCM 字节实时写入 `mpv` 的 `stdin`
6. `mpv` 以 `rawaudio` 模式播放

关键细节：

1. 不是等完整音频生成后再播放，而是边生成边播
2. `Base64StreamDecoder` 解决了“base64 被拆在多个 chunk 里”的问题
3. `PCMFrameBuffer` 会丢弃尾部不完整的 PCM 帧，避免 `rawaudio` 播放时出现结尾噪音
4. 工具调用轮次的中间音频不会播放，避免工具轮短噪声或半成品音频泄露给用户
5. 播放器关闭前会补一个小的静音尾巴，避免最后一个字的尾音被切掉
6. 播放速度由配置控制，当前默认做了轻微加速

## 4.8 UI 与交互体验

终端交互在 [ui.py](/home/howard/ai-workspace/mini-agent/src/voice/ui.py)。

主要能力：

1. `PromptToolkit` 提供命令历史
2. `Rich` 欢迎面板展示模型、音色、工作目录与使用提示
3. `Rich Live` 面板实时显示：
   - 当前状态
   - 最近工具调用
   - 当前答案文本
   - usage 摘要

这让 `VoiceAgent` 不只是一个“会播声音的脚本”，而是一个可交互、可调试、可观察的语音代理终端。

## 5. 关键实现亮点

## 5.1 真正的全模态链路

这里不是“语音只是输入壳子、真正逻辑还是纯文本代理”。

核心模型在整个链路中直接负责：

1. 理解语音输入
2. 决策是否需要调用工具
3. 基于工具结果生成最终文本
4. 生成最终语音输出

## 5.2 工具调用后仍然由 Qwen 输出最终声音

传统代理系统常见做法是：

1. 工具执行完
2. 拼一段文本答案
3. 再交给 TTS 播放

本项目不是这样。

本项目中：

1. 工具只负责提供事实或执行动作
2. 最终面向用户的文本和语音仍然由 `Qwen3.5-Omni` 统一生成

这使得工具结果融入自然语言和语音风格时更一致。

## 5.3 流式音频解码与即时播放

很多“多模态 demo”只做到：

1. 请求模型
2. 拿到完整音频
3. 保存文件
4. 再播放

当前项目已经做到：

1. 流式接收音频 chunk
2. 增量 base64 解码
3. 实时喂给播放器

这比“整段生成后再播”更接近真正语音助手体验。

## 6. 当前配置项

配置已集中在 [config.py](/home/howard/ai-workspace/mini-agent/src/voice/config.py)。

常用环境变量包括：

1. `DASHSCOPE_API_KEY`
2. `DASHSCOPE_BASE_URL`
3. `QWEN_OMNI_MODEL`
4. `QWEN_OMNI_VOICE`
5. `QWEN_ENABLE_SEARCH`
6. `VOICE_AGENT_MAX_HISTORY_TURNS`
7. `VOICE_AGENT_INPUT_SAMPLE_RATE`
8. `VOICE_AGENT_PCM_SAMPLE_RATE`
9. `VOICE_AGENT_PCM_CHANNELS`
10. `VOICE_AGENT_PCM_SAMPLE_WIDTH`
11. `VOICE_AGENT_TAIL_PADDING_MS`
12. `VOICE_AGENT_PLAYBACK_SPEED`

## 7. 关键测试覆盖

当前测试文件是 [test_voice_agent.py](/home/howard/ai-workspace/mini-agent/tests/test_voice_agent.py)。

已经覆盖：

1. base64 分片解码
2. PCM 不完整尾帧丢弃
3. 音频输入消息构造
4. 多轮历史裁剪
5. tool call 参数分片拼接
6. 一轮流式响应中的文本、音频、usage 处理
7. tool call 触发后继续下一轮生成最终答复
8. tool call 轮次出现时停止当前音频播放

这意味着当前 VoiceAgent 的核心行为已经不只是“人工跑起来看效果”，而是有自动化测试保护。

## 8. 总结

`VoiceAgent` 的核心价值不只是“能录音和播音”，而是它采用了一个更接近端到端多模态代理的架构：

1. 输入端直接接受语音或文本
2. 推理端由 `Qwen3.5-Omni` 统一处理理解、工具决策和最终答复生成
3. 输出端直接播放模型返回的流式音频

相比传统 `ASR + LLM + TTS` 方案，这种架构在工程整合、模态一致性、实时交互体验和后续多模态扩展上都有明显优势。  
而通过当前项目中的 `tool use + 流式音频 + PromptToolkit + Rich + 配置抽离 + 测试覆盖`，它已经不是一个简单 demo，而是一套可维护、可扩展的语音代理实现。
