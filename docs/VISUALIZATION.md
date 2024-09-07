### Result Visualization

#### 3D Object Detection
```bash
torchpack dist-run -np 1 python tools/visualize.py [config file] --checkpoint [checkpoint file] --out-dir [output dir] --mode [pred/gt] --bbox-score 0.1 
```
> Notics: All arguments other than bbox-score are quite self-explanatory, for the bbox-score, it is a threshold filtering out low-confidence predictions. You can select this value according to the visual quality of your generated figures. A higher threshold will remove more duplicate predictions and create a cleaner visualization. Please refer to [visualize](https://github.com/mit-han-lab/bevfusion/issues/211) for more details.

#### BEV Map Segmentation
```bash
torchpack dist-run -np 1 python tools/visualize.py [config file] --checkpoint [checkpoint file] --out-dir [output dir] --mode [pred/gt] --map-score 0.5 
```

> Tip: Notice that the generated visualization files might be a little bit large (~50G) if you visualize for both 3D object detection and bev map segmentation results.

> Please have a look at tools/visualize.py before you run the code for visualization.

> Tip: You can refer to the visualization tutorial of [nuScences](https://www.nuscenes.org/nuscenes?tutorial=maps) website for more details.
