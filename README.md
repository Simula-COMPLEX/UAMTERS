# UAMTERS: Uncertainty-Aware Mutation Analysis for DL-enabled Robotic Software


> To facilitate reviewing our proposed approach, reviewers please refer to the corresponding data in this repository.<br/>

This repository contains:

1. **[deepqtest-project](https://github.com/Simula-COMPLEX/DeepQTest/tree/main/deepqtest-project)** - all related algorithms and source code for conducting the experiment;
2. **[formal-experiment](https://github.com/Simula-COMPLEX/DeepQTest/tree/main/formal-experiment)** - all the raw data for the experiment results and analyses;
3. **[rest-api](https://github.com/Simula-COMPLEX/DeepQTest/tree/main/rest-api)** - the list of REST API endpoints for configuring the environment.

## Overview

 Self-adaptive robots adjust their behaviors in response to unpredictable environmental changes. These robots often incorporate deep learning (DL) components into their software to support functionality such as perception, decision-making, and control, enhancing autonomy and self-adaptability. However, the inherent uncertainty of DL-enabled software makes it challenging to ensure its dependability in dynamic environments. Consequently, test generation techniques have been developed to test robot software, and classical mutation analysis injects faults into the software to assess the test suite's effectiveness in detecting the resulting failures. However, there is a lack of mutation analysis techniques to assess the effectiveness under the uncertainty inherent to DL-enabled software. To this end, we propose UAMTERS, an uncertainty-aware mutation analysis framework that introduces uncertainty-aware mutation operators to explicitly inject stochastic uncertainty into DL-enabled robotic software, simulating uncertainty in its behavior. We further propose mutation score metrics to quantify a test suite's ability to detect failures under varying levels of uncertainty. We evaluate UAMTERS across three robotic case studies, demonstrating that UAMTERS more effectively distinguishes test suite quality and captures uncertainty-induced failures in DL-enabled software.

<!-- ## UAMTERS Overview -->

 <div align=center><img src="https://github.com/Simula-COMPLEX/UAMTERS/blob/main/figs/overview.png" width="960" /></div>

