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
	unzip -n $ROOT/$(basename $LINK) -d $ROOT
done
rm $ROOT/val2014/COCO_val2014_000000320612.jpg

python3 cache.py -m cache/datasets=cache.voc.cache cache/name=cache_voc cache/category=config/category/20

echo test models with 20 classes
python3 cache.py -m cache/datasets='cache.voc.cache cache.coco.cache' cache/name=cache_20 cache/category=config/category/20

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
python3 cache.py -m cache/datasets=cache.coco.cache cache/name=cache_80 cache/category=config/category/80

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

wget http://pjreddie.com/media/files/darknet19_448.conv.23 -nc -P ~/model/darknet

export CACHE_NAME=cache_voc MODEL_NAME=model_voc MODEL=model.yolo2.Darknet
python3 convert_darknet_torch.py ~/model/darknet/darknet19_448.conv.23 -m model/name=$MODEL_NAME model/dnn=$MODEL -d
python3 train.py -b 64 -lr 1e-3 -e 160 -m cache/name=$CACHE_NAME model/name=$MODEL_NAME model/dnn=$MODEL train/optimizer='lambda params, lr: torch.optim.SGD(params, lr, weight_decay=5e-4, momentum=0.9)' train/scheduler='lambda optimizer: torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 90], gamma=0.1)'
python3 eval.py -m cache/name=$CACHE_NAME model/name=$MODEL_NAME model/dnn=$MODEL
