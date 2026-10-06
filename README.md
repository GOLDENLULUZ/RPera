# RPera

RPera 是一个在本机运行的 AI 角色扮演冒险应用。选择世界、模组和开局场景后，输入想做的事；多个 AI Agent 会按需协作，生成可以继续游玩的故事。界面为中文，支持桌面和移动端浏览器。

## 1.1.0 更新

- 支持 Windows 10/11 原生运行：双击 `start.bat` 启动、`update.bat` 更新，无需安装 WSL。
- 玩家上传的图片向 Agent 提供精确存档路径；角色设计 Agent 可直接复用原图作为立绘，无需开启绘画能力或重新生成图片。
- 实体、报告和工具结果以保留原文的结构化文本交给模型，避免 JSON 转义影响阅读与精确编辑；支持原会话继续和回合 retry。
- 后台故事摘要按回合显示，摘要编辑的原文、新文本及完整 JSON 可折叠查看。
- 合规性审核 Agent 不再提供主动实体读取工具，报告读取能力保持可用。

## 开始使用

需要 [uv](https://docs.astral.sh/uv/getting-started/installation/) 和一个受支持的 AI 服务的 API Key。`uv` 会在启动时准备 Python 3.12+ 和项目依赖；普通用户不需要构建 wheel 或安装 Node.js。

### Windows 10/11

在 PowerShell 中安装 uv：

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

安装后重新打开终端。获取项目可使用 [Git for Windows](https://git-scm.com/download/win)：

```powershell
git clone https://github.com/GOLDENLULUZ/RPera.git
cd RPera
.\start.bat
```

也可以在 GitHub 仓库页面选择「Code → Download ZIP」，解压后双击根目录的 `start.bat`。脚本检查 uv，按锁文件准备运行环境，并读取 `config/server.json` 的端口和监听配置；默认在浏览器打开 <http://127.0.0.1:8000>。首次启动需联网下载依赖，启动失败时窗口会保留错误信息。项目应放在本地可写目录，支持中文及空格路径；Windows 和 WSL 不要共用同一个 `.venv/`。

### Linux / WSL

运行：

```bash
git clone https://github.com/GOLDENLULUZ/RPera.git
cd RPera
./start.sh
```

打开 <http://127.0.0.1:8000>，然后：

1. 在「设置 → AI 模型」中新建 AI Preset，填写服务地址、API Key 和模型名称；在「通用」中将它分配给主代理。其他 Agent 默认跟随主代理，也可以分别指定 Preset。
2. 在「冒险」中选择示例世界「雾港」，可以再选择模组「读心大师」和开局场景，创建存档。
3. 输入角色的行动，等待故事生成；在「前台」继续冒险，在「后台」查看运行记录和存档实体。

支持 DeepSeek、OpenAI-compatible、Google Gemini、Anthropic Claude 和 xAI Grok。OpenAI-compatible 与 Grok 可以按连接选择 Chat Completions 或 Responses API；还可以配置一个共享备用 AI，在首选模型遇到内容拦截或临时服务错误时额外尝试一次。AI 服务可能按其自身规则收费；没有可用的模型连接时，无法生成故事。结束运行请在终端按 `Ctrl+C`。

### Windows 更新

先关闭正在运行的 RPera，再双击 `update.bat`。自动更新需要 Git for Windows、uv，以及通过 Git 克隆的项目；脚本调用 `scripts/update.ps1`，先获取当前分支的上游更新，再以 `git merge --ff-only --no-overwrite-ignore` 合并并同步依赖。

本地源码、Prompt 或其他未忽略文件有修改时会停止，不自动覆盖、stash 或 reset。新版本文件若与被忽略的个人配置、存档或自建内容同路径，也会停止而不覆盖。若依赖安装失败，源码可能已经更新，修正问题后运行 `start.bat` 即可。

ZIP 下载不支持自动更新。将新版解压到另一个目录，再复制旧版的 `config/`、`saves/` 和自建世界、模组、文风；自改 Prompt 需手动合并，不要直接覆盖整份旧目录或复制 `.venv/`。

## 能做什么

- 创建独立存档，选择参与模式、叙事人称和语言；从已完成的回合创建分支。
- 让专业 Agent 按需总结故事、检索世界设定、维护角色、地点与目标、提供角色意见、规划文风、撰写与检查故事。
- 上传玩家图片，并按需直接用作角色立绘；配置 Stable Diffusion WebUI 或 NovelAI 后，可以选择启用原创角色立绘生成。
- 使用可验证的实体与报告引用在 Agent 间传递上下文，并在后台查看安全排版的完整运行记录。
- 在「创作」中编辑世界、模组、场景、实体和文风；在「后台」查看执行记录与当前存档的实体。
- 通过 ZIP 分享、导入和导出世界或模组；同名导入自动追加 `(1)`、`(2)`。
- 中止正在运行的回合，并在最后一个回合失败或中断后继续执行。

仓库只附带示例世界「雾港」、模组「读心大师」和文风「冷峻悬疑」。可以在「创作」中添加自己的内容。合规性审核 Agent 默认调用所分配的 AI；也可以在通用设置中切换为仓库内可编辑的固定响应。

### 分享世界与模组

在「创作」的世界或模组分类点击「导入 ZIP」，或者选中来源后点击「导出 ZIP」。每个包只包含一个 `worlds/<名称>/` 或 `mods/<名称>/`，保留现有的 `description.md`、`scenarios/*.md` 和 `entities/<类型>/<名称>/ENTITY.md`，不需要 manifest；空白来源也可以导出。手工制作的 ZIP 使用相同结构即可导入，包内类型必须与当前分类一致。

导入名称取自包内目录，同名时自动依次追加 `(1)`、`(2)` 并采用第一个可用名称；原名称已有编号时继续追加，超长时截短基础名称。导入成功后自动选中新内容。ZIP 最大 16 MiB、解压内容最大 64 MiB、最多 4096 个文件和目录条目。无效包完整拒绝，不留下部分内容。

导出只包含已保存的源内容，保留文件原字节；页面尚未保存的编辑不会包含在内。源内容变化不影响已有存档自己的快照。

### 模型思考强度

AI Preset 按型号提供思考强度选单。已识别的 Gemini 3 型号显示各自支持的档位；DeepSeek 的 `deepseek-flash`、`deepseek-v4-pro`、`deepseek-v4-flash` 和 `deepseek-v4-flash-vision-exp` 支持关闭／低／高／最大。默认「自动」不发送思考参数；其他型号仍可使用自动模式。新建 DeepSeek Preset 默认使用 `deepseek-flash`，已有配置保持原值。

## 本地数据和配置

默认情况下，个人配置保存在项目根目录的 `config/`，存档保存在 `saves/`；它们不应提交到 Git。可以设置 `RPERA_USER_DATA_DIR`，将配置和存档放到其他目录。世界、模组和文风放在项目的 `data/` 下；新增内容如需公开，请自行检查后再提交。

服务默认只监听本机的 `127.0.0.1:8000`，无需创建配置文件。要修改端口或启用局域网访问，在用户数据目录新建 `config/server.json`，例如：

```json
{
  "lan_access": false,
  "port": 8000,
  "basic_auth": {
    "enabled": false,
    "username": "",
    "password": ""
  }
}
```

更改后重启 `start.bat`（Windows）或 `./start.sh`（Linux / WSL）。`lan_access: true` 会监听 `0.0.0.0`；如需设置访问密码，请同时将 `basic_auth.enabled` 改为 `true`，并填写用户名和密码。HTTP Basic 在普通 HTTP 下不会加密密码；跨不可信网络访问请使用 HTTPS。Windows 原生运行需允许对应端口通过防火墙；WSL2 下还可能需要配置 Windows 到 WSL 的端口转发。

开发者可以运行 `uv run pytest` 验证服务端与公开 HTTP 闭环。后台 Markdown 渲染的源文件位于 `web/trace-markdown.js`；修改后在 `web/` 运行 `npm ci && npm run build:trace` 生成浏览器直接使用的 `web/static/trace-markdown.js`，并用 `npm test` 验证渲染。正常运行 RPera 不需要安装 Node.js。源码使用 Python 3.12+、FastAPI、SQLite，以及少量原生 JavaScript。

## 许可证

本项目采用 [MIT 许可证](LICENSE)。
