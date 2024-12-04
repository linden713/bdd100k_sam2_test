# Project Overview

This project focuses on applying **SAM2** for autonomous driving instance segmentation tasks using the **BDD100K dataset**. The objective is to implement a model capable of detecting and segmenting objects from the dataset in real time.

---
<center class ='img'>
<img title="Visual Result 1" src="https://github.com/linden713/bdd100k_sam2_test/blob/main/visual_result/masked_7d06fefd-f7be05a6.jpg" width="45%">
<img title="Visual Result 2" src="https://github.com/linden713/bdd100k_sam2_test/blob/main/visual_result/masked_7d97d173-09388af3.jpg" width="45%">
</center>

## Usage

1. **Download SAM2**  
   Follow the instructions on the SAM2 repository to download and set up the framework:  
   [https://github.com/facebookresearch/sam2](https://github.com/facebookresearch/sam2)

2. **Download BDD100K Dataset and Annotations**  
   - Dataset and annotations download: [https://github.com/bdd100k/bdd100k](https://github.com/bdd100k/bdd100k)  
   - BDD100k download: [https://doc.bdd100k.com/download.html#](https://doc.bdd100k.com/download.html#)

3. **Convert Dataset to COCO Format**  
   Use the following command to convert BDD100K instance segmentation data to COCO format:
   ```bash
   python3 -m bdd100k.label.to_coco -m ins_seg \
       -i ${in_path} -o ${out_path} [--nproc ${process_num}]
   ```
   Replace in_path with the input path, out_path with the desired output path, and ${process_num} with the number of processes to use.
4. **Update Annotation Path**
   
   Update the annotation path in the bdd100k.py script to point to your dataset annotations.
5. **Run the Script**
    ```bash
    python bdd100k.py
    ```


## File Structure

Below is an overview of the project's directory structure, detailing the purpose of each file and folder:
```plaintext
  project/
  ├── bdd100k_ins_seg_labels_trainval/
  │   └── bdd100k/
  │       └── labels/
  │           └── ins_seg/
  │               ├── bitmasks/
  │               ├── colormaps/
  │               ├── polygons/
  │               ├── rles/
  │               └── val/
  ├── result/
  │   ├── base_plus/
  │   │   ├── bdd100k_val.json
  │   │   └── result.txt
  │   ├── large/
  │   ├── small/
  │   └── tiny/
  ├── visual_result/
  ├── bdd100k.py
  ├── calculate_iou.py
  ├── gen_predict_mask.py
  └── pycoco.py
```

### `bdd100k_ins_seg_labels_trainval/bdd100k/` 
Note: The dataset is too large to include in the repository; you need to download the dataset yourself.

Contains essential files and labels for training and validation using the BDD100K dataset.

- **`labels/ins_seg/`**  
  This folder includes the labels for instance segmentation, such as:
  - `bitmasks/`: Stores bitmask images for each object instance.
  - `colormaps/`: Contains colormap files that are used for visualizing the segmentation results.
  - `polygons/`: Includes polygon-based segmentation annotations.
  - `rles/`: Run-length encoding data for segmentation masks.
  - `val/`: Validation data used for model evaluation.

### `result/`
This folder stores the results generated during the testing phase.

- **`base_plus/`**
  Contains results using the base_plus model weights:
  - `bdd100k_val.json`: A JSON file with the validation set annotations.
  - `result.txt`: A text file that logs the results and outcomes of tests.

- **`large/`**, **`small/`**, **`tiny/`**  
  These directories organize results based on model size, with different configurations of the test data and predictions.



### `visual_result/`

  Stores the visual results of the sam2 large model pt, including segmentation output images.

### Python Scripts

- **`bdd100k.py`**  
  This script processes images and uses the SAM2 model to predict segmentation masks, converts them into COCO-compatible format, and saves the predictions to a JSON file for evaluation.

- **`calculate_lou.py`**  
  Contains functions for calculating metrics like Intersection over Union (IoU) to evaluate segmentation performance.

- **`gen_predict_mask.py`**  
  A script for generating and saving the predicted segmentation masks based on the model's output.

- **`pycoco.py`**  
  A script for generating AP & AR.
## Summary

This project leverages the BDD100K dataset to perform instance segmentation for autonomous driving, utilizing the SAM2 model to handle the task effectively. The structure of the project is organized to streamline training, validation, testing, and result visualization.

## MIT Lisence
