# Chapter 9: Using PyTorch to fight cancer

### Main points
1. This section of the book will be about lung tumor detection: take three-dimensional CT scans of human torsos as input, and produce the location of suspected malignant tumors, if any, as output
2. CT scan voxels are rectangular prism shapes (as opposed to voxels in other applications which are usually cubes) because of the way the CT scanner measures along the different axis, and each voxel has a numeric value that roughly corresponds to the average mass density of the matter contained inside
3. The steps will be: 1. load raw CT data into usable form → 2. segment voxels of potential tumors (i.e. produce heatmap of areas to be fed to classifier) → 3. group interesting voxels into nodules (won't use PyTorch) → 4. classify as nodules or non-nodules → 5. classify nodules as benign or malignant

### 9.1 Introduction to the use case

- This section of the book will be about lung tumor detection: take three-dimensional CT scans of human torsos as input, and produce the location of suspected malignant tumors, if any, as output
- This task is difficult for humans because usually there is usually no cancer in the cases, but we want to catch the cases where there is → it is comparable to being placed in front of 100 haystacks and being told, “Determine which of these, if any, contain a needle.”
- Automating this process will give experience working with complicated data and develop problem solving skills

### 9.2 Preparing for a large-scale project

- Since this project involves non-standard data, a lot of the work will involve data manipulation outside of the classification model environment
- We are also going to simplify the whole problem into a series of smaller problems
- Learning about the problem space (in this case, radiation oncology) is crucial

### 9.3 What is a CT scan, exactly?

- CT scans are essentially X-rays, represented as a 3D array of single-channel data (like a stack of greyscale png images)
- A voxel (volumetric pixel) is the 3D equivalent of a 2D pixel
- CT scan voxels are rectangular prism shapes (as opposed to voxels in other applications which are usually cubes) because of the way the CT scanner measures along the different axis, and each voxel has a numeric value that roughly corresponds to the average mass density of the matter contained inside

### 9.4 The project: An end-to-end detector for lung cancer

- The steps will be: 1. load raw CT data into usable form → 2. segment voxels of potential tumors (i.e. produce heatmap of areas to be fed to classifier) → 3. group interesting voxels into nodules (won't use PyTorch) → 4. classify as nodules or non-nodules → 5. classify nodules as benign or malignant
- Even though we might be tempted to implement an end-to-end model for detection and classification, those models are typically trained on hundreds of thousands of images and the positive (malignant nodules) results aren't as rare as in our case
- Our datasource will be from the LUNA (LUng Nodule Analysis) Grand Challenge
