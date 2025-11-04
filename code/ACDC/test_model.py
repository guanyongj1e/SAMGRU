#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os
import time
import logging
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import ACDCdataset
from utils import test_single_volume
from modelGRU import SAGRU
from encoder import MTUNet

def setup_logger(log_file):
    """Configure logging"""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Create file handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    
    # Create console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    # Attach handlers to logger
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger

def inference(args, model, testloader, logger, test_save_path=None):
    """Run inference and compute evaluation metrics"""
    logger.info(f"Starting inference on {len(testloader)} test samples")
    model.eval()
    metric_list = 0.0
    with torch.no_grad():
        for i_batch, sampled_batch in tqdm(enumerate(testloader)):
            h, w = sampled_batch["image"].size()[2:]
            image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
            metric_i = test_single_volume(image, label, model, classes=args.num_classes, patch_size=[args.img_size, args.img_size],
                                          test_save_path=test_save_path, case=case_name, z_spacing=args.z_spacing)
            metric_list += np.array(metric_i)
            logger.info(f'Sample {i_batch} case {case_name} mean Dice {np.mean(metric_i, axis=0)[0]:.4f} mean HD95 {np.mean(metric_i, axis=0)[1]:.4f}')
        
        metric_list = metric_list / len(testloader)
        for i in range(1, args.num_classes):
            logger.info(f'Class {i} mean Dice {metric_list[i-1][0]:.4f} mean HD95 {metric_list[i-1][1]:.4f}')
        
        performance = np.mean(metric_list, axis=0)[0]
        mean_hd95 = np.mean(metric_list, axis=0)[1]
        logger.info(f'Overall test performance: mean Dice: {performance:.4f} mean HD95: {mean_hd95:.4f}')
        logger.info("Testing completed!")
        
        return performance, mean_hd95

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to model weights file")
    parser.add_argument("--model_type", type=str, default="SAGRU", help="Model type")
    parser.add_argument("--volume_path", type=str, default="./data/ACDC/test", help="Test data directory")
    parser.add_argument("--list_dir", type=str, default="./data/ACDC/lists_ACDC", help="List directory for dataset splits")
    parser.add_argument("--img_size", type=int, default=224, help="Image size")
    parser.add_argument("--num_classes", type=int, default=4, help="Number of classes")
    parser.add_argument("--z_spacing", type=int, default=10, help="Z-axis spacing")
    parser.add_argument("--test_save_dir", type=str, default="./predictions", help="Directory to save predictions")
    parser.add_argument("--gpu", type=str, default="0", help="GPU id to use")
    args = parser.parse_args()
    
    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    # Create output directory
    if not os.path.exists(args.test_save_dir):
        os.makedirs(args.test_save_dir)
    
    # Configure logging
    log_file = os.path.join(args.test_save_dir, f"test_{args.model_type}_{time.strftime('%Y%m%d_%H%M%S')}.log")
    logger = setup_logger(log_file)
    
    # Log test configuration
    logger.info(f"Model type: {args.model_type}")
    logger.info(f"Model path: {args.model_path}")
    logger.info(f"Test data path: {args.volume_path}")
    logger.info(f"Image size: {args.img_size}")
    logger.info(f"Save directory: {args.test_save_dir}")
    
    # Load dataset
    logger.info("Loading test dataset...")
    db_test = ACDCdataset(base_dir=args.volume_path, list_dir=args.list_dir, split="test")
    testloader = DataLoader(db_test, batch_size=1, shuffle=False)
    logger.info(f"Test dataset contains {len(db_test)} samples")
    
    # Create model
    logger.info(f"Creating {args.model_type} model...")
    encoder = MTUNet(64)
    

    if args.model_type == 'SAGRU':
        model = SAGRU(num_classes=args.num_classes, attenion_size=88*88, encoder=encoder)

    
    # Load model weights
    logger.info(f"Loading model weights: {args.model_path}")
    model.load_state_dict(torch.load(args.model_path))
    model = model.cuda()
    
    # Run test
    logger.info("Start testing...")
    start_time = time.time()
    performance, mean_hd95 = inference(args, model, testloader, logger, args.test_save_dir)
    test_time = (time.time() - start_time) / 60
    logger.info(f"Testing finished! Elapsed: {test_time:.2f} minutes")
    logger.info(f"Overall performance: Dice: {performance:.4f}, HD95: {mean_hd95:.4f}")
    logger.info(f"Segmentation results saved to: {args.test_save_dir}")
    
    # Return results for further processing
    return {
        "dice": performance,
        "hd95": mean_hd95,
        "save_dir": args.test_save_dir
    }

if __name__ == "__main__":
    results = main() 