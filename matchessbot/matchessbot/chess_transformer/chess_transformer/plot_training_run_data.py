import os

import numpy as np

import matplotlib.pyplot as plt

import csv

from config import import_config



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

    CONFIG = import_config(model_config_name="CT-E-20", run_number=7)

    n_epochs = 50

    logs_dir = CONFIG['LOGS_FOLDER'] + "/training_data_tensorboard_csv"
    filename_prefix = "run-run_7-tag-"
    filename_postfix = ".csv"
    # tf_event_filename = "events.out.tfevents.1740382723.chainsaw.166148.0"


    # all_train_log_tags_step = ["train_loss", "train_lr", "train_data_time", "train_step_time", "train/top1_accuracy", "train/top3_accuracy", "train/top5_accuracy"]
    all_train_log_tags_epoch = ["train_epoch_avg_loss", "train_epoch_avg_accuracy_top1", "train_epoch_avg_accuracy_top3", "train_epoch_avg_accuracy_top5"]
    all_train_log_tags_valid_epoch = ["val_loss", "val_top1_accuracy", "val_top3_accuracy", "val_top5_accuracy"]

    headers_file = []
    
    train_tag = all_train_log_tags_epoch[0]
    valid_tag = all_train_log_tags_valid_epoch[0]


    filename_train = filename_prefix + train_tag + filename_postfix
    epochs, avg_train_loss = read_csv_training_data(log_dir=logs_dir, filename=filename_train)

    filename_valid = filename_prefix + valid_tag + filename_postfix
    epochs, avg_valid_loss = read_csv_training_data(log_dir=logs_dir, filename=filename_valid)

    # print(len(epochs[:n_epochs]))
    
    plt.figure()
    plt.plot(epochs[:n_epochs],avg_train_loss[:n_epochs], epochs[:n_epochs], avg_valid_loss[:n_epochs])
    plt.ylabel('Loss',fontsize="x-large")
    plt.xlabel('Epoch',fontsize="x-large")
    plt.legend(['Training Loss', 'Validation Loss'],fontsize="large")
    plt.axis([0, 51, 0, 7.5])
    plt.grid()
    save_file = os.path.join(logs_dir, 'plot_train_valid_epoch_loss.png')
    plt.savefig(save_file)
    # plt.show()



    train_acc_tags = all_train_log_tags_epoch[1:]
    valid_acc_tags = all_train_log_tags_valid_epoch[1:]

    # for train_acc_tag in train_acc_tags:
    #     filename_train = filename_prefix + train_acc_tag + filename_postfix
    #     epochs, avg_train_acc_top_k = read_csv_training_data(log_dir=logs_dir, filename=filename_train)
    #     plt.plot(epochs[:n_epochs], avg_train_acc_top_k[:n_epochs])

    # for valid_acc_tag in valid_acc_tags:
    #     filename_train = filename_prefix + valid_acc_tag + filename_postfix
    #     epochs, avg_train_acc_top_k = read_csv_training_data(log_dir=logs_dir, filename=filename_train)
    #     plt.plot(epochs[:n_epochs], avg_train_acc_top_k[:n_epochs])

    # plt.ylabel('Accuracy')
    # plt.xlabel('Epoch')
    # plt.legend(['Train Acc. Top1', 'Valid Acc. Top1', 'Train Acc. Top3', 'Valid Acc. Top3', 'Train Acc. Top5', 'Valid Acc. Top5'])
    # plt.show()

    accuracy_names = ['Top1', 'Top3', 'Top5']

    idx = 0

    for train_acc_tag, valid_acc_tag in zip(train_acc_tags, valid_acc_tags):
        filename_train = filename_prefix + train_acc_tag + filename_postfix
        epochs_train, avg_train_acc_top_k = read_csv_training_data(log_dir=logs_dir, filename=filename_train)

        filename_valid = filename_prefix + valid_acc_tag + filename_postfix
        epochs_valid, avg_valid_acc_top_k = read_csv_training_data(log_dir=logs_dir, filename=filename_valid)
        plt.figure()
        plt.plot(epochs_train[:n_epochs], avg_train_acc_top_k[:n_epochs], epochs_valid[:n_epochs], avg_valid_acc_top_k[:n_epochs])
        plt.axis([0, 51, 0, 1.1])
        plt.ylabel('Accuracy',fontsize="x-large")
        plt.xlabel('Epoch',fontsize="x-large")
        # plt.legend(['Train Acc. ' + accuracy_names[idx], 'Valid Acc. '  + accuracy_names[idx]],fontsize="large")
        plt.legend(['Train ' + accuracy_names[idx], 'Valid '  + accuracy_names[idx]],fontsize="large")
        plt.grid()
        save_file = os.path.join(logs_dir, 'plot_train_valid_epoch_acc_' + accuracy_names[idx] + '.png')
        plt.savefig(save_file)
        # plt.show()
        idx += 1