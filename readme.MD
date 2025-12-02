# BINDGAR is Interface for N-body Data, especially for GENGA And REBOUND.

## Declaration / 声明

### License / 许可证
This project is licensed under the **GNU General Public License v3.0 (GPLv3)**.

本项目采用 **GNU通用公共许可证v3.0（GPLv3）** 发布。

**All commercial uses are strictly prohibited**, including but not limited to:

**明确禁止所有商业用途**，包括但不限于：
- Commercial product sales or services / 商业产品销售或服务
- Enterprise for-profit projects / 企业盈利性项目
- Any form of commercial application / 任何形式的商业应用
- **Especially** / **特别严禁用于**
    - Repackaging and redistribution on social media platforms as "fan benefits" or "resource packages" that require following, adding friends, or other forms of engagement to access / 在社交媒体上重新打包传播，作为需要关注、加好友等条件才能获取的"粉丝福利"或"资源整合包"
    - Marketing accounts redistributing the software for traffic generation, follower growth, or other promotional purposes / 营销号以引流、涨粉等为目的的软件传播行为
    - Any form of paid or restricted-access redistribution / 任何形式的付费或限条件传播行为
    - Bundling this software into commercial packages or value-added services / 将本软件捆绑进商业套餐或增值服务中

Academic research, educational institutions, and personal learning are fully permitted without special authorization.

学术研究、教育机构和个人学习使用完全开放，无需特殊授权。

### Important Disclaimer / 重要免责声明
This software package was developed with the assistance of generative AI technology and **has not undergone comprehensive, rigorous testing and verification**.

本软件包是在生成式人工智能技术辅助下开发的，**未经全面、严谨的测试与验证**。

**This software is provided "as is", without warranty of any kind**, express or implied. Users assume all risks associated with the use of this software. The author shall not be liable for any loss or damage arising from the use of this software.

**本软件按"现状"提供，不附带任何明示或暗示的担保**。使用者应自行承担使用本软件的所有风险，作者对因使用本软件而产生的任何损失或损害不承担责任。

### Special Notice / 特别提示
- **Must** independently verify and test the critical functionalities of this software / **请务必**独立的验证和测试本软件的关键功能
- Problem reports and suggestions are welcome, but timely fixes cannot be guaranteed / 欢迎反馈问题和建议，但无法保证及时修复


# Description

**Bindgar** is an interface for N-body data, especially for [GENGA](https://bitbucket.org/sigrimm/gengas) and [REBOUND](https://github.com/hannorein/rebound). The name is a recursive acronym. 

**BINDGAR bridges these two powerful tools**:

- **REBOUND** excels at setting up initial conditions and rapid simulation prototyping
- **GENGA** specializes in high-precision, long-term planetary system evolution
It allows you to seamlessly transition your REBOUND simulations into GENGA's high-precision integration environment.

**Warning**: The author hasn't bothered uploading this to PyPI yet. You'll have to install it manually, by cloning the repository and running `pip install .` in the project directory. And add an "-e" flag if you want to edit the code later.

**Warning**: The author hasn't bothered writing documentation yet. But please use `bindgar -h` to see the help message.

**Warning**: The author hasn't bothered writing tests yet. Use at your own risk.

**Warning**: The author hasn't bothered making this cross-platform yet. It is only tested on Linux and MacOS.

**Warning**: It is an "on-the-fly" project. The author may add features or change the API at any time without notice.


# 中文介绍
Bindgar 是一个专为 N 体数据打造的接口工具，特别为 [GENGA](https://bitbucket.org/sigrimm/gengas) 和 [REBOUND](https://github.com/hannorein/rebound). 这两款神器量身定制！而且哦，它的名字其实是个递归缩写～

**Bindgar** 为你架起一座无缝桥梁：

- **REBOUND**：一个十分便利的N体模拟python库，只是实际计算时算起来有点慢
- **GENGA**：一个超级无敌强大的N体模拟软件，专注于高精度、长期的行星系统演化模拟，还可以利用GPU加速计算！只不过GENGA 的输入输出格式有点复杂

**Bindgar** 主要利用REBOUND来设置初始条件，然后转化成GENGA的输入文件，还能分析GENGA的输出结果。

**碎碎念**：作者懒得把它上传到PyPI，所以你得手动克隆代码库，然后在项目目录下运行 `pip install .` 来安装。如果你想以后修改代码的话，记得加上 `-e` 标志。

**碎碎念**：作者懒得写专门的文档，不过为了防止作者自己忘了怎么用，还是在代码里加了点帮助信息，试着运行 `bindgar -h`。

**碎碎念**：作者懒得写测试代码，所以请自行承担使用风险。

**碎碎念**：作者懒得进行跨平台测试，所以目前只在 Linux 和 MacOS 上测试过。

**碎碎念**：这是作者为了便于自己入门 N-body 模拟而造的轮子，可能随着项目进展随时添加功能或更改 API。