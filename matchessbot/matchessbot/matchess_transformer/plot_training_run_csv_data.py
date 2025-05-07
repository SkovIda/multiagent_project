import os

import numpy as np

import matplotlib.pyplot as plt

import csv

# from configs import import_config



def read_csv_training_data(log_dir, filename):
    filepath = os.path.join(log_dir, filename)
    
    row_count = 0
    epochs = []
    data = []
    with open(filepath, 'r') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')#, quotechar='|')
        for row in csvreader:
            if row_count == 0:
                headers_file = row
                # print(headers_file)
            else:
                epochs.append(int(row[1]))
                data.append(float(row[2]))
            row_count += 1
    return epochs, data





if __name__ == '__main__':

    # Get configuration

    # CONFIG = import_config(model_config_name="CT-E-20", run_number=7)

    n_epochs = 150

    result_save_filename_prefix = 'epoch_150'

    MATChessFormer_Homogeneous_training_loss_csv_filepath = "./results/run_1/tensorboard_training_data" + "_" + result_save_filename_prefix

    csv_filename_prefix = "run-run_1-tag-"
    filename_postfix = ".csv"
    result_save_dir = "./results/run_1/figs"
    
    csv_logs_dir = MATChessFormer_Homogeneous_training_loss_csv_filepath

    all_train_log_tags_epoch = ["train_epoch_avg_loss", "train_epoch_avg_accuracy_top1", "train_epoch_avg_accuracy_top3", "train_epoch_avg_accuracy_top5"]
    all_train_log_tags_valid_epoch = ["val_loss", "val_top1_accuracy", "val_top3_accuracy", "val_top5_accuracy"]

    loss_train_log_tag_idx = 0
    train_loss_filename = csv_filename_prefix + all_train_log_tags_epoch[loss_train_log_tag_idx] + filename_postfix
    val_loss_filename = csv_filename_prefix + all_train_log_tags_valid_epoch[loss_train_log_tag_idx] + filename_postfix

    epochs_train, avg_train_loss = read_csv_training_data(log_dir=csv_logs_dir, filename=train_loss_filename)
    epochs_val, avg_val_loss = read_csv_training_data(log_dir=csv_logs_dir, filename=val_loss_filename)

    plt.figure(1)
    plt.axis([0, n_epochs + 1, 0, 8.1])
    plt.plot(epochs_train[:n_epochs], avg_train_loss[:n_epochs], epochs_val[:n_epochs],avg_val_loss[:n_epochs], linewidth=2)
    plt.legend(['Train Loss', 'Valid Loss'])
    plt.title('Mean Label-smoothed Cross-Entropy Loss\nof the Homogeneous Agents',fontsize="x-large", fontweight='bold')
    plt.ylabel('Loss',fontsize="large", fontweight='bold')
    plt.xlabel('Epoch',fontsize="large", fontweight='bold')
    plt.grid()

    save_file = os.path.join(result_save_dir, result_save_filename_prefix + '_plot_train_valid_loss.png')
    plt.savefig(save_file)

    ################################

    train_acc_tags_encoder = all_train_log_tags_epoch[1:]
    valid_acc_tags_encoder = all_train_log_tags_valid_epoch[1:]

    plt.figure(2)
    plt.axis([0, n_epochs + 1, 0, 1.1])
    for train_acc_tag in train_acc_tags_encoder:
        filename_train = csv_filename_prefix + train_acc_tag + filename_postfix
        epochs, avg_train_acc_top_k = read_csv_training_data(log_dir=csv_logs_dir, filename=filename_train)
        plt.plot(epochs[:n_epochs], avg_train_acc_top_k[:n_epochs])

    for valid_acc_tag in valid_acc_tags_encoder:
        filename_train = csv_filename_prefix + valid_acc_tag + filename_postfix
        epochs, avg_train_acc_top_k = read_csv_training_data(log_dir=csv_logs_dir, filename=filename_train)
        plt.plot(epochs[:n_epochs], avg_train_acc_top_k[:n_epochs])

    plt.legend(['Train top1', 'Train top3', 'Train top5', 'Valid top1', 'Valid top3', 'Valid top5'])#,fontsize="large")
    plt.title('Average Top-k Accuracy of the Homogeneous Agents',fontsize="x-large", fontweight='bold')
    plt.ylabel('Accuracy',fontsize="large", fontweight='bold')
    plt.xlabel('Epoch',fontsize="large", fontweight='bold')
    plt.grid()

    save_file = os.path.join(result_save_dir, result_save_filename_prefix + '_plot_train_valid_topk_accuracy.png')
    plt.savefig(save_file)
