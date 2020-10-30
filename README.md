# Part-Localization-by-Hypercolumn

## Requirement
* 資料集CUB200-2011，請到[此處](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html)下載
* 如果您有使用Docker，這裡有已經安裝好的環境[jchuang0710/hypercolumn](https://hub.docker.com/repository/docker/jchuang0710/hypercolumn)，或是使用指令`docker push jchuang0710/hypercolumn`
## Path Setting
* 將Setting.py中的CUB_FOLDER_PATH 設為您下載CUB200-2011，所放置的位置
* 如果您要測試，請將Probability.ipynb中 Detector所載入模型的位置修改成您訓練模型的位置
## Training
* 由於模型為40MB，因此不方便上傳
* 將CreatModel.ipynb打開並執行，可以在裡面選擇想訓練的地方
## Testing
* 將Probability.ipynb打開並執行
## RESULT
 
|IoU>0.5 (%/s)|bbox|body|head|
|----------|----|-----|----|
|with Mask||||
|增加訓練的資料|||0.644/0.25|
