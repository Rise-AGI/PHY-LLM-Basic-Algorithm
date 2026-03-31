# TypeScript（实验流程可视化工具）

## 零基础含义解释
TypeScript就像**物理实验的流程可视化工具**，在JavaScript基础上增加类型检查，确保实验步骤的规范性。

## 物理场景类比解释
在粒子物理实验中：
- TypeScript相当于**实验数据动态展示板**：
  - 类型检查类似实验仪器的量程限制（如粒子能量检测范围）
  - 接口定义实验数据格式（类似探测器信号传输标准）
  - 模块化实验流程（类似分阶段数据处理管道）

## 基本配置
### Windows/macOS/Linux三平台安装教程
1. **Windows系统**
   - 安装Node.js：https://nodejs.org/dist/v20.11.0/node-v20.11.0-x64.msi
   - 双击安装包 → 勾选"Add to PATH" → 点击"Next"
   - 验证Node.js：`node -v`（终端输入）
   - 安装TypeScript：`npm install -g typescript`

2. **macOS系统**
   - 安装Homebrew：`/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"`
   - 安装Node.js：`brew install node@20`
   - 安装TypeScript：`npm install -g typescript`

3. **Linux系统（Ubuntu）**
   - 安装Node.js：`sudo apt install nodejs npm`
   - 验证安装：`node -v`
   - 安装TypeScript：`npm install -g typescript`

```bash
# 运行环境：Node.js 20.11 + TypeScript 5.3 + VSCode TypeScript插件
# 物理实验数据处理项目初始化
npm init -y  # 创建package.json配置文件
npm install --save typescript  # 将TS添加为项目依赖
```

## 物理系学生常见应用场景
- 实验数据格式标准化（如粒子轨迹JSON结构）
- 探测器控制接口定义（如ADC/DAC设备规范）
- 实验流程模块化（如数据采集/分析分离）

## 算法应用（物理实验场景）
```ts
// 运行环境：Node.js 20.11 + TypeScript 5.3 + VSCode物理数据插件
// 量子态可视化与光谱数据处理
import { writeFileSync } from 'fs';  // 文件系统模块（存储实验数据）

// 定义光谱数据接口（类似探测器信号规范）
interface SpectrumData {
    wavelengths: number[];  // 波长数组（nm）
    intensities: number[];  // 强度值（counts）
    metadata: {
        experimentId: string;  // 实验编号
        timestamp: Date;       // 采集时间
    }
}

// 创建量子态叠加可视化模块
class QuantumStateVisualizer {
    private amplitude: number;
    private phase: number;

    constructor(amplitude: number, phase: number) {
        this.amplitude = amplitude;
        this.phase = phase;
    }

    // 生成波函数数据（模拟量子叠加态）
    generateWaveFunction(steps: number): {x: number[], y: number[]} {
        const result = {x: [], y: []};
        for(let i=0; i<steps; i++) {
            const x = i * 0.1;
            result.x.push(x);
            result.y.push(this.amplitude * Math.sin(x + this.phase));
        }
        return result;
    }
}

// 物理实验数据保存函数
function saveExperimentData(data: SpectrumData): void {
    const jsonData = JSON.stringify(data, null, 2);  // 格式化JSON
    writeFileSync('实验数据.json', jsonData);  // 保存到文件
    console.log('数据已保存至实验数据.json');
}

// 使用示例
const visualizer = new QuantumStateVisualizer(1.5, 0.5);
const waveData = visualizer.generateWaveFunction(100);
saveExperimentData({
    wavelengths: [400, 500, 600],
    intensities: [120, 200, 80],
    metadata: {
        experimentId: "Q-EXP-2024-001",
        timestamp: new Date()
    }
});
```

### 代码中文注释说明
1. **物理意义注释**：
   - `SpectrumData`接口：定义光谱仪采集数据的标准格式
   - `QuantumStateVisualizer`类：模拟量子态叠加的波形生成

2. **实验场景注释**：
   - `generateWaveFunction`：生成量子态波函数数据，用于可视化分析
   - `saveExperimentData`：标准化保存物理实验数据，确保可追溯性

## VSCode操作指引
1. **安装插件**
   - 打开VSCode → Ctrl+Shift+X → 搜索"TypeScript" → 确保内置插件已启用
   - 安装"JavaScript and TypeScript Nightly"插件（增强物理数据处理）

2. **创建TS配置文件**
   - Ctrl+Shift+P → 输入"TypeScript: Create tsconfig.json"
   - 修改配置文件内容：
```json
{
  "compilerOptions": {
    "target": "ES2022",  // 设置ECMAScript版本
    "module": "NodeNext",  // 支持Node.js模块系统
    "strict": true,      // 启用严格类型检查（类似实验校准）
    "outDir": "./dist"   // 指定输出目录
  },
  "include": ["src/**/*"]  // 包含物理实验代码目录
}
```

3. **调试物理代码**
   - 创建launch.json配置：
```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "type": "pwa-node",
      "request": "launch",
      "name": "调试物理实验代码",
      "runtimeExecutable": "nodemon",
      "restart": true,
      "console": "integratedTerminal",
      "internalConsoleOptions": "neverOpen",
      "program": "${workspaceFolder}/src/index.ts"
    }
  ]
}
```
   - 按F5启动调试 → 观察量子态波形数据生成过程
