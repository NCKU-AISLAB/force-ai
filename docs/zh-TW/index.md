---
hide:
  - navigation
  - toc
---

# FORCE AI

**FORCE AI** (Fast Optimization for Resource-Constrained and Efficient AI Inference) 是一個專為教育目的打造的開放原始碼框架，展示如何從零開始設計一個針對特定模型進行效能優化的 AI 加速器。專案內容涵蓋從模型訓練與量化、效能分析、硬體架構設計，到編譯器與執行階段的完整流程，幫助你理解並實作 end-to-end AI 推論加速。

本框架以可教學、可擴充為核心精神，設計簡潔、模組化，適合用於課程實作、自學專案、或作為進階研究的基礎起點。所有工具與模組皆基於開源技術構建，包括 PyTorch、TVM、Verilator 與 ONNX，並搭配自行設計的 power-of-2 symmetric quantization 以及 Python-based performance model，結合 RTL 模擬與硬體參數探索，實現以設計為導向的 AI 加速器開發。

## :mortar_board: 適合對象

- AI 加速器領域的初學者、研究生、大學高年級生
- 希望學習如何串接 AI inference 與硬體設計的開發者
- 想以小成本建立屬於自己的加速器實驗環境者

## :wrench: 你會學到

- 如何訓練並量化神經網路模型以利硬體部署
- 如何設計與模擬基於 Eyeriss 的 CNN 加速器
- 如何使用 Verilator 做 RTL-level 模擬與驗證
- 如何建構簡易 inference runtime 並使用 TVM 編譯器進行 code generation
- 如何從前端模型需求導出適當的硬體設計規格
