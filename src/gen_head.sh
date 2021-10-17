# echo "MOT"
# python crop_head_bbox.py --full ../../dataset/MOT20/labels_with_ids_full_body/ --image ../../dataset/MOT20/images/ --out ../../dataset/MOT20/labels_with_ids
# python crop_head_bbox.py --full ../../dataset/MOT17/labels_with_ids_full_body/ --image ../../dataset/MOT17/images/ --out ../../dataset/MOT17/labels_with_ids
# python crop_head_bbox.py --full ../../dataset/MOT16/labels_with_ids_full_body/ --image ../../dataset/MOT16/images/ --out ../../dataset/MOT16/labels_with_ids
# python crop_head_bbox.py --full ../../dataset/MOT15/labels_with_ids_full_body/ --image ../../dataset/MOT15/images/ --out ../../dataset/MOT15/labels_with_ids
# echo "ETHZ"
# python crop_head_bbox.py --full ../../dataset/ETHZ/eth01/labels_with_ids_full_body/ --image ../../dataset/ETHZ/eth01/images/ --out ../../dataset/ETHZ/eth01/labels_with_ids --image_ext png
# python crop_head_bbox.py --full ../../dataset/ETHZ/eth02/labels_with_ids_full_body/ --image ../../dataset/ETHZ/eth02/images/ --out ../../dataset/ETHZ/eth02/labels_with_ids --image_ext png
# python crop_head_bbox.py --full ../../dataset/ETHZ/eth03/labels_with_ids_full_body/ --image ../../dataset/ETHZ/eth03/images/ --out ../../dataset/ETHZ/eth03/labels_with_ids --image_ext png
# python crop_head_bbox.py --full ../../dataset/ETHZ/eth05/labels_with_ids_full_body/ --image ../../dataset/ETHZ/eth05/images/ --out ../../dataset/ETHZ/eth05/labels_with_ids --image_ext png
# python crop_head_bbox.py --full ../../dataset/ETHZ/eth07/labels_with_ids_full_body/ --image ../../dataset/ETHZ/eth07/images/ --out ../../dataset/ETHZ/eth07/labels_with_ids --image_ext png
# echo "CUHK-SYSU"
# python crop_head_bbox.py --full ../../dataset/CUHK-SYSU/labels_with_ids_full_body/ --image ../../dataset/CUHK-SYSU/images/ --out ../../dataset/CUHK-SYSU/labels_with_ids

# echo "Citypersons"
# python crop_head_bbox.py --full ../../dataset/Citypersons/labels_with_ids_full_body/ --image ../../dataset/Citypersons/images/ --out ../../dataset/Citypersons/labels_with_ids --image_ext png

# echo "PRW"
# python crop_head_bbox.py --full ../../dataset/PRW/labels_with_ids_full_body/ --image ../../dataset/PRW/images/ --out ../../dataset/PRW/labels_with_ids

echo "Caltech"
python crop_head_bbox.py --full ../../dataset/CalTech/labels_with_ids_full_body/ --image ../../dataset/CalTech/images/ --out ../../dataset/CalTech/labels_with_ids --image_ext png
