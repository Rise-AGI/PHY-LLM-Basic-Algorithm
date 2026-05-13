- 新建train/data文件夹, 存储Success、Faile的任务的本地监视时的所有print()的字符串之和(当然也要有[]和换行), 保存为一个文件, 用'提交时间(比如202604261952)+本地提交任务的py文件简写(比如warmup_packages写为wp, 同理dma,ss)+magnus任务名'命名, 这份文件由monitor自动生成(似乎之后需要把任务提交者的py名也传入monitor)
- 新建一份文件来记录我提交的服务器长期存储成功(由warm-up和model-download的py自动记录), 分类(目前只有modelscope,pip,model-version )按时间排序 ; 注意model-version用于记录成功生成报告的模型的magnus和本地地址 ,采用模型名-v1, 模型名-v2命名, 需要submit-SFT如果成功生成报告, 则立即下载到train/SFT_data文件夹中, 命名任然名字-v1
- 如果job的名字已经存在于model-version中(比如Qwen-2.5-35b-v1已存在), 则拒绝此次提交
- 整理md笔记梳理train文件夹各个py的功能和接口信息





- 我删除了data, 新建data1 ,data2 分别存储日志和时间线同名文件, 后缀.data1 / .data2
让日志和时间线文件第一行写一个类似latex bib参考文献的格式
`@{time =
name =
submitter =
....
job唯一编号 
类型(日志/时间线) 
status = 

等等数据(尽量给全部参数, 因为为了使得新旧版本data通用, {}内参数不会改动)

}`
如果程序在文件夹中识别到非标准文件名, 则exe自动读取{} 确认文件是否重复, 然后选择重命名文件





train/download_model_auto.py服务器上运行失败:
(下载成功日志省略)
下载完成，临时路径: /tmp/model_download
=== 移动到持久目录 ===
cp: error writing '/data/magnus/models/Qwen2.5-Math-PRM-7B/model-00001-of-00004.safetensors': No space left on device
cp: error writing '/data/magnus/models/Qwen2.5-Math-PRM-7B/model-00002-of-00004.safetensors': No space left on device
cp: error writing '/data/magnus/models/Qwen2.5-Math-PRM-7B/model-00003-of-00004.safetensors': No space left on device
cp: error writing '/data/magnus/models/Qwen2.5-Math-PRM-7B/model-00004-of-00004.safetensors': No space left on device
cp: error writing '/data/magnus/models/Qwen2.5-Math-PRM-7B/model.safetensors.index.json': No space left on device
cp: error writing '/data/magnus/models/Qwen2.5-Math-PRM-7B/modeling_qwen2_rm.py': No space left on device
cp: error writing '/data/magnus/models/Qwen2.5-Math-PRM-7B/tokenizer_config.json': No space left on device
cp: error writing '/data/magnus/models/Qwen2.5-Math-PRM-7B/vocab.json': No space left on device


Installing collected packages: xxhash, shellingham, sentencepiece, safetensors, regex, python-dateutil, pyarrow, protobuf, propcache, multidict, mdurl, hf-xet, h11, frozenlist, einops, dill, click, anyio, annotated-doc, aiohappyeyeballs, yarl, pandas, multiprocess, markdown-it-py, httpcore, aiosignal, rich, httpx, aiohttp, typer, huggingface-hub, tokenizers, datasets, accelerate, transformers
  Attempting uninstall: click
    Found existing installation: click 8.1.7
    Uninstalling click-8.1.7:
      Successfully uninstalled click-8.1.7
这里卡了很久

待办:
验证SFT微调
整理已经创建的程序和笔记
MoEW微调蓝图
RF微调蓝图

## 260506待办
- 为SFT训练集和测试集的api添加统一调整提示词的蓝图入口[Y]
- 设计提示词[Y]
- 修改train\inspect_storage.py, 查询几条完整的训练数据[Y]
- 优化SFT[Y]
- 自动评价系统
- 72B模型
- 单卡多卡验证[Y]
- 服务器部署大模型
