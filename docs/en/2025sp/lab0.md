# Lab 0.1 - Working Environment Setup

## Overview

這門課主要會用到 Python、C/C++、Verilog，以及些許的 Linux Shell 命令，所有的 lab 作業將會以 Markdown 格式進行撰寫，因此在這個 lab 當中我們會介紹如何設置和使用這門課所需要的開發環境，並幫大家複習 Python、C/C++、Verilog programming、如何使用開源的 Verilator 撰寫 testbench，以及如何瀏覽和編輯 Markdown 文件。

這份 lab 不列入成績計算，也不需要繳交作業，但會有一些練習幫助大家快速熟悉上述的開發工具。

## AISLab Server

請同學填寫 [AOC2025 課程帳號註冊表單](https://forms.gle/Hdrd7hhtnHmvoQAT6)，收到 email 取得帳號密碼後，嘗試登入 ais01 伺服器

注意：由於資訊安全考量，本伺服器資源僅限透過校內 IP 使用，需要在成大網域內 (連學校 Wi-Fi) 才能登入，若在校外可使用成大計網中心提供的 [SSL VPN](https://cc.ncku.edu.tw/p/412-1213-7637.php?Lang=zh-tw) 服務，請依照連結內的說明安裝並使用成功入口帳號登入

### Login via SSH

開啟終端機 (macOS 的 [Terminal](https://support.apple.com/zh-tw/guide/terminal/apd5265185d-f365-44cb-8b09-71a064a42125/mac) 或 Windows 的 [Powershell](https://learn.microsoft.com/zh-tw/powershell/scripting/overview?view=powershell-7.4)/[Command Prompt](https://opensourcedoc.com/windows-programming/command-prompt-primer/))，輸入以下命令使用 [Secure Shell (SSH)](https://zh.wikipedia.org/zh-tw/Secure_Shell) 登入 AISLab Server，`$STUDENT_ID` 的部分請改成你自己的學號 (英文字母請用小寫)，接著會要求輸入密碼，請用 email 中取得的密碼登入

```bash
ssh $STUDENT_ID@ais01.ee.ncku.edu.tw
```
:::spoiler 登入成功畫面
<center><img width=70% src="https://hackmd.io/_uploads/Sk1XjMM_yx.png" /></center><br>
:::

:::spoiler 登入失敗解法
若登入失敗，顯示錯誤訊息為 `Permission denied ...`，可能是帳號或密碼輸入錯誤，請檢查英文字母大小寫，注意學號前面的 `$` 要拿掉，例如：

```shell
ssh a12345678@ais01.ee.ncku.edu.tw
```

或是另外宣告成變數，例如：

```shell
STUDENT_ID=a12345678
ssh $STUDENT_ID@ais01.ee.ncku.edu.tw
```

若出現其他錯誤訊息，請於 Discord 的 **lab0-discussion-board** 貼文發問，<font color=red>**得到答案後請不要刪文**</font>
:::

<br>

伺服器使用 Ubuntu 20.04，為 Linux 作業系統的其中一個發行版 (distribution)，使用預設的殼層 [Bash](https://zh.wikipedia.org/zh-tw/Bash)，終端機命令的使用同學可以自行上網搜尋使用教學、詢問 ChatGPT，或參考以下連結：

- [The Linux command line for beginners | Ubuntu](https://ubuntu.com/tutorials/command-line-for-beginners#1-overview)
- [Bash Scripting Tutorial – Linux Shell Script and Command Line for Beginners](https://www.freecodecamp.org/news/bash-scripting-tutorial-linux-shell-script-and-command-line-for-beginners/)
- [簡明 Linux Shell Script 入門教學 | TechBridge 技術共筆部落格](https://blog.techbridge.cc/2019/11/15/linux-shell-script-tutorial/)

### Login via VSCode Remote SSH Plugin

在終端機環境中雖然有 Vim 或 Nano 等文字編輯器可以使用，但學習成本較高，因此更常見的做法是使用 Visual Studio Code (VSCode) 遠端連線進行程式碼的撰寫與開發

![image](https://hackmd.io/_uploads/HJpcGXfuke.png)

安裝和設定的方式請參考官方文件：[Developing on Remote Machines using SSH and Visual Studio Code](https://code.visualstudio.com/docs/remote/ssh)

<!-- 首先在 VSCode 中安裝 [Remote SSH](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-ssh) 擴充套件，接著按 Ctrl+Shift+P (macOS 為 Command+Shift+P) -->

### Change Password

成功登入後請輸入以下命令更改密碼

```bash
passwd
```

依照提示先填寫 Email 中取得的初始密碼，再輸入新的密碼

### SSH Key (Optional)

每次登入都需要輸入密碼很麻煩，可以使用 SSH Key 的方式做驗證，免去輸入密碼的麻煩，請在**本地端**輸入以下命令：

```bash
ssh-keygen -t ed25519
```

接下來會提示使用者輸入一些資訊，若希望維持預設，則直接按 <kbd>Enter</kbd>；若有自訂 ssh key 路徑，下面指令需要注意使用對的路徑

然後用以下命令將本地端產生的公鑰傳送並儲存在伺服器上，其中 `$STUDENT_ID` 填寫自己的學號，英文字母請用小寫

```bash
# for macOS and Linux users
ssh-copy-id $STUDENT_ID@ais01.ee.ncku.edu.tw

# for Windows users
cat ~/.ssh/id_ed25519.pub | ssh $STUDENT_ID@ais01.ee.ncku.edu.tw "mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys"
```

成功後就可以免密碼登入

```bash
ssh $STUDENT_ID@ais01.ee.ncku.edu.tw
```

### SSH Config (Optional)

在**本地端**新增 `~/.ssh/config`，若該檔案已存在，則開啟檔案新增以下內容，`$STUDENT_ID` 填寫自己的學號，英文字母請用小寫

```bash
Host aoc2025
    HostName ais01.ee.ncku.edu.tw
    User $STUDENT_ID
    Port 22
    IdentityFile ~/.ssh/id_ed25519
```

完成後即可用更簡潔的名稱登入遠端伺服器

```bash
ssh aoc2025
```

### FAQ

若登入上有遇到問題，可以參考 [Lab 0 - FAQ](/VhIZSjsUSZujDmyJ78-bgw)，或在 Discord 的 **lab0-discussion-board** 上貼文，或在 Discord 上私訊助教

發問時請詳細描述相關資訊，讓助教可以快速辨識問題發生的原因並提供解法，例如：

- 使用的作業系統環境 (e.g. Windows, macOS, Linux, WSL, Docker, etc.)
- 做了什麼操作
- 預期結果
- 實際結果或錯誤訊息

<font color=red>**注意：在討論區貼文發問，得到答案後請不要刪文**</font>

## Python Virtual Environment & Package Manager

### Install Miniconda

登入伺服器後，請在終端機輸入以下指令自[官網](https://docs.anaconda.com/miniconda/install/#quick-command-line-install)下載 Miniconda3

```bash
[[ ! -f /tmp/miniconda.sh ]] && wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
bash /tmp/miniconda.sh -b -u -p ~/miniconda3
source ~/miniconda3/bin/activate
conda init --all
```

安裝完成後我們需要建立一個環境，在終端機輸入以下指令即可建立虛擬環境，以下以 `test-env` 為例，使用 Python 3.10，下方命令中 `-n` 為環境名稱，`-y` 表示自動安裝，不需要手動確認

```bash
# conda create -n (你想要的環境名稱) python=(你想要的版本號)
conda create -n test-env python=3.10 -y
```

稍等片刻後等待套件安裝完畢，使用指令可確認現有環境

```bash
conda env list
```

建立完成後可使用以下指令開啟或關閉環境:

```bash
conda activate test-env # 開啟環境
conda deactivate        # 關閉環境
```

環境使用完後若有刪除環境的需求，請於關閉環境後輸入以下指令

```bash
# conda remove --name (環境名稱) --all
conda remove --name test-env --all
```

開啟環境後可以安裝你所需要的套件

```bash
conda install pandas
```

安裝過程中若有反應不如預期的情況，歡迎在 Discord 對應的頻道上發問

### PIP

除了 `conda` 指令，也可以使用 `pip` 指令安裝套件

```bash
pip install pandas
```

助教提供 `requirements.txt` 時，請使用以下指令安裝所需套件，`conda` 指令不支援這部分

```bash
pip install -r requirements.txt
```

### venv (Optional)

venv 為 Python 內建的虛擬環境管理工具，由於本課程主要使用 Conda，因此僅提供 [官方文件](https://docs.python.org/3.13/library/venv.html) 連結，若同學有興趣可自行嘗試

## Markdown

### Introduction

[Markdown](https://zh.wikipedia.org/zh-tw/Markdown) 是一個輕量化的標記語言，常用於技術文件的撰寫，可以透過簡單的語法來表示文件呈現的格式，大幅簡化文件撰寫時的格式調整，讓開發者可以專注在內容的產出。這門課的每個 lab 我們都將會使用 Markdown 格式來呈現講義的內容，同學也需要使用 Markdown 來撰寫作業的報告，下面提供一些參考資料介紹如何編輯和瀏覽 Markdown 文件。

<!-- [Markdown](https://en.wikipedia.org/wiki/Markdown) is a lightweight markup language commonly used for technical document writing. It allows formatting of documents through simple syntax, significantly simplifying the adjustment of formatting during the writing process. This enables developers to focus on content creation. -->

<!-- In this course, every lab will use Markdown format to present lecture notes, and students are also required to use Markdown to write their assignment reports. Below are some reference materials introducing how to edit and view Markdown documents. -->

### Local Markdown Viewer/Editor

在本地端可以使用以下工具來編輯和瀏覽 Markdown 文件：

- [VSCode](https://code.visualstudio.com/docs/languages/markdown) (built-in, no plugin installation needed)
- [MarkText](https://www.marktext.cc/)
- [Obsidian](https://obsidian.md/)
<!-- - Web Browser plugins
    - Chrome/Edge/Arc:
    - Safari:
    - Firefox:  -->

### Useful VSCode Extensions for Markdown

以下是一些在 VSCode 中撰寫 Markdown 文件時常用的擴充套件：

- [Markdownlint](https://marketplace.visualstudio.com/items?itemName=DavidAnson.vscode-markdownlint): Markdown linting and style checking for Visual Studio Code
- [Markdown PDF](https://marketplace.visualstudio.com/items?itemName=yzane.markdown-pdf): Convert Markdown to PDF/HTML/PNG/JPEG

### Markdown Syntax

若同學對 Markdown 語法不熟悉，可以參考以下連結：

參考：[HackMD 使用教學](https://hackmd.io/c/tutorials-tw/%2Fs%2Fbasic-markdown-formatting-tw)
