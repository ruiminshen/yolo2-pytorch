echo download VOC dataset
LINKS="
http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
"
ROOT=~/data
for LINK in $LINKS
do
	wget $LINK -nc -P $ROOT
	tar -kxvf $ROOT/$(basename $LINK) -C $ROOT
done

echo download COCO dataset
LINKS="
http://images.cocodataset.org/zips/train2014.zip
http://images.cocodataset.org/zips/val2014.zip
http://images.cocodataset.org/annotations/annotations_trainval2014.zip
http://images.cocodataset.org/zips/train2017.zip
http://images.cocodataset.org/zips/val2017.zip
http://images.cocodataset.org/annotations/annotations_trainval2017.zip
"
ROOT=~/data/coco
for LINK in $LINKS
do
	wget $LINK -nc -P $ROOT
	tar -kxvf $ROOT/$(basename $LINK) -C $ROOT
done
rm $ROOT/val2014/COCO_val2014_000000320612.jpg

python3 cache.py -m cache/datasets=voc cache/name=cache_voc cache/category=config/category/20

echo test models with 20 classes
python3 cache.py -m cache/datasets="voc coco" cache/name=cache_20 cache/category=config/category/20

MODELS="
yolo-voc
tiny-yolo-voc
"

for MODEL in $MODELS
do
	wget http://pjreddie.com/media/files/$MODEL.weights -nc -P ~/model/darknet
	python3 convert_darknet_torch.py ~/model/darknet/$MODEL.weights -c config.ini config/darknet/$MODEL.ini -d
	python3 eval.py -c config.ini config/darknet/$MODEL.ini -m cache/name=cache_20
	python3 detect.py -c config.ini config/darknet/$MODEL.ini -i image.jpg --pause
done

echo test models with 80 classes
python3 cache.py -m cache/datasets="voc coco" cache/name=cache_80 cache/category=config/category/80

MODELS="
yolo
"

for MODEL in $MODELS
do
	wget http://pjreddie.com/media/files/$MODEL.weights -nc -P ~/model/darknet
	python3 convert_darknet_torch.py ~/model/darknet/$MODEL.weights -c config.ini config/darknet/$MODEL.ini -d
	python3 eval.py -c config.ini config/darknet/$MODEL.ini -m cache/name=cache_80
	python3 detect.py -c config.ini config/darknet/$MODEL.ini -i image.jpg --pause
done
