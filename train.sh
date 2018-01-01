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

exit

python3 cache.py -m cache/datasets="voc coco" cache/name=cache_20

MODELS="
darknet19_448.conv.23
"

for MODEL in $MODELS
do
	wget http://pjreddie.com/media/files/$MODEL.weights -nc -P ~/model/darknet
	python3 convert_darknet_model.py ~/model/darknet/$MODEL.weights -c config.ini config/darknet.ini -d
done
