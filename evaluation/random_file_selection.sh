# Taken from: https://stackoverflow.com/a/414316
# Change hardcoded dirs and num_files

source_folder=/datasets/train2017/ # MS COCO image folder
dest_folder=../logs/ablation_study/content_imgs
num_files=20

ls $source_folder |sort -R |tail -$num_files |while read file; do
    # Something involving $file, or you can leave
    # off the while to just get the filenames
    cp $source_folder/$file $dest_folder/$file
done
