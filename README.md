# Instructions

1. Create .zip with files from this repo
2. Create a new project in AI Centre
3. Upload ML Package with the .zip file create on step 1
	- Input type: file
	- Language: Python 3.8 OPENCV
	- Enable Training: True
4. Create dataset and make sure it has the following folder structure:
	- MyDataset (Dataset Folder)
		- Training (Folder)
			- Name1 (Folder) - Place photos from Name1 here in PNG format
			- Name2 (Folder) - Place photos from Name1 here in PNG format
			- Name3 (Folder) - Place photos from Name1 here in PNG format
			- …
		- Evaluation (Folder) 
			- Name1 (Folder) - Place photos from Name1 here in PNG format
			- Name2 (Folder) - Place photos from Name1 here in PNG format
			- Name3 (Folder) - Place photos from Name1 here in PNG format
			- …
4. Create a new training pipeline with your dataset
	- Make sure to select the Training folder in the input dataset field
5. Create an ML Skill after the package has been trained
