real=1
fake=0
spatialThresholdDegree=5

inputRealDir="/home/yangchihyuan/anaconda3/dataset/all_dataset/"
inputFakeDir="/home/yangchihyuan/anaconda3/dataset/attack_dataset/"
outputDir="/home/yangchihyuan/anaconda3/dataset/phase3/train/"

[ ! -d "$outputDir" ] && mkdir -p "$outputDir"

# process real dataset
python make_face_dataset.py --mode $real --inputDir $inputRealDir --outputDir $outputDir --spatialThresholdDegree $spatialThresholdDegree

# process fake dataset
python make_face_dataset.py --mode $fake --inputDir $inputFakeDir --outputDir $outputDir --spatialThresholdDegree $spatialThresholdDegree
