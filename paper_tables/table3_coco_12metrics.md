# Table 3: Full COCO 12-Metrics Suite

| Metric Symbol | Metric Description | Simple Route | Full OrchestraNet | Gain ($\Delta$) |
| :--- | :--- | :---: | :---: | :---: |
| **$AP$** | Primary Challenge Metric (IoU=0.50:0.95) | 0.4848 | **0.4883** | **+0.35%** |
| **$AP_{50}$** | Standard PASCAL VOC Metric (IoU=0.50) | 0.6516 | **0.6500** | **-0.16%** |
| **$AP_{75}$** | Strict Localization Accuracy (IoU=0.75) | 0.5236 | **0.5321** | **+0.85%** |
| **$AP_S$** | Small Objects (Area $< 32^2$) | 0.3313 | **0.3312** | **-0.01%** |
| **$AP_M$** | Medium Objects ($32^2 < \text{Area} < 96^2$) | 0.6136 | 0.6122 | -0.14% |
| **$AP_L$** | Large Objects (Area $> 96^2$) | 0.7541 | **0.7548** | **+0.07%** |
| **$AR_1$** | Average Recall with 1 detection/image | 0.4564 | **0.4557** | **-0.08%** |
| **$AR_{10}$** | Average Recall with 10 detections/image | 0.7742 | **0.7735** | **-0.07%** |
| **$AR_{100}$** | Average Recall with 100 detections/image | 0.8389 | **0.8395** | **+0.06%** |
| **$AR_S$** | Small Object Recall (Area $< 32^2$) | 0.6304 | **0.6323** | **+0.19%** |
| **$AR_M$** | Medium Object Recall ($32^2 < \text{Area} < 96^2$) | 0.8759 | 0.8755 | -0.05% |
| **$AR_L$** | Large Object Recall (Area $> 96^2$) | 0.8948 | **0.8953** | **+0.05%** |
