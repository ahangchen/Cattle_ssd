ffmpeg -i VID_20180101_153937.mp4 -r 10 -f image2 %05d.jpg
export PYTHONPATH=$PYTHONPATH:/home/cwh/coding/models/research/slim:/home/cwh/coding/Cattle_ssd/object_detection
python /home/cwh/coding/Cattle_ssd/create_cow_record.py
rm ssd_fine_tune/graph.pbtxt ssd_fine_tune/checkpoint ssd_fine_tune/events.out.tfevents.* ssd_fine_tune/model.ckpt-*
python /home/cwh/coding/Cattle_ssd/object_detection/train.py --logtostderr --pipeline_config_path=/home/cwh/coding/Cattle_ssd/model/ssd_mobilenet_v1_coco.config --train_dir=/home/cwh/coding/Cattle_ssd/data
rm -rf model/frozen/*
python /home/cwh/coding/Cattle_ssd/object_detection/export_inference_graph.py --input_type image_tensor --pipeline_config_path /home/cwh/coding/Cattle_ssd/model/ssd_mobilenet_v1_coco.config --trained_checkpoint_prefix /home/cwh/coding/Cattle_ssd/model/model.ckpt-1000 --output_directory /home/cwh/coding/Cattle_ssd/model/frozen
python /home/cwh/coding/Cattle_ssd/count_by_video.py
ffmpeg -r 10 -f image2 -i %05d.jpg -vcodec libx264 -crf 25 -pix_fmt yuv420p detect.mp4