data/hyps/hyp.scratch-low.yaml              
data/nia_basketball.yaml      
mask_yolo_ann_txt.py                        
json_ls.txt                    
####mv_no_location.sh & mv_val_split.sh


####sbatch_yolo_basketball.sh. 
(for multi-gpus)  
(yolov5-s default == 1 gpu, batch size 64, lr 0.01 -> 4 gpus * 24 bs per gpu, lr 0.015 (linear-scaling)  
'''
python -m torch.distributed.run --nproc_per_node 4 \  
    train.py --img 1920 --batch 96 --epochs 300 --data nia_basketball.yaml --weights yolov5s.pt --device 0,1,2,3
'''
