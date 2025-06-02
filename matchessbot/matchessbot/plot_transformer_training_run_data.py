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

    n_epochs = 50

    logs_dir_decoder = "./chessformers/training_data/logs/run_5/" + "training_data_tensorboard_csv"
    filename_prefix_decoder = "run-run_5-tag-"
    filename_postfix = ".csv"

    # all_train_log_tags_step = ["train_loss", "train_lr", "train_data_time", "train_step_time", "train/top1_accuracy", "train/top3_accuracy", "train/top5_accuracy"]
    all_train_log_tags_epoch_decoder = ["train_epoch_loss", "train_epoch_avg_accuracy_top1", "train_epoch_avg_accuracy_top3", "train_epoch_avg_accuracy_top5"]
    all_train_log_tags_valid_epoch_decoder = ["val_loss", "val_top1_accuracy", "val_top3_accuracy", "val_top5_accuracy"]

    headers_file_decoder = []
    
    train_tag_decoder = all_train_log_tags_epoch_decoder[0]
    valid_tag_decoder = all_train_log_tags_valid_epoch_decoder[0]


    filename_train_decoder = filename_prefix_decoder + train_tag_decoder + filename_postfix
    epochs_train_decoder, avg_train_loss_decoder = read_csv_training_data(log_dir=logs_dir_decoder, filename=filename_train_decoder)

    filename_valid_decoder = filename_prefix_decoder + valid_tag_decoder + filename_postfix
    epochs_val_decoder, avg_valid_loss_decoder = read_csv_training_data(log_dir=logs_dir_decoder, filename=filename_valid_decoder)

    # print(len(epochs[:n_epochs]))
    
    logs_dir_encoder = "./chess_transformer/chess_transformer/training_runs/CT-E-20/logs/run_7/" + "training_data_tensorboard_csv"
    filename_prefix_encoder = "run-run_7-tag-"
    filename_postfix = ".csv"

    # all_train_log_tags_step = ["train_loss", "train_lr", "train_data_time", "train_step_time", "train/top1_accuracy", "train/top3_accuracy", "train/top5_accuracy"]
    all_train_log_tags_epoch_encoder = ["train_epoch_avg_loss", "train_epoch_avg_accuracy_top1", "train_epoch_avg_accuracy_top3", "train_epoch_avg_accuracy_top5"]
    all_train_log_tags_valid_epoch_encoder = ["val_loss", "val_top1_accuracy", "val_top3_accuracy", "val_top5_accuracy"]

    headers_file_encoder = []
    
    train_tag_encoder = all_train_log_tags_epoch_encoder[0]
    valid_tag_encoder = all_train_log_tags_valid_epoch_encoder[0]


    filename_train_encoder = filename_prefix_encoder + train_tag_encoder + filename_postfix
    epochs_train_encoder, avg_train_loss_encoder = read_csv_training_data(log_dir=logs_dir_encoder, filename=filename_train_encoder)

    filename_valid_encoder = filename_prefix_encoder + valid_tag_encoder + filename_postfix
    epochs_val_encoder, avg_valid_loss_encoder = read_csv_training_data(log_dir=logs_dir_encoder, filename=filename_valid_encoder)


    result_save_dir = "./test_results"

    # plt.figure(1)
    # plt.plot(epochs_train_decoder[:n_epochs],avg_train_loss_decoder[:n_epochs], epochs_train_decoder[:n_epochs], avg_valid_loss_decoder[:n_epochs], epochs_train_encoder[:n_epochs], avg_train_loss_encoder[:n_epochs], epochs_val_encoder[:n_epochs],avg_valid_loss_encoder[:n_epochs], linewidth=2)
    # plt.ylabel('Loss',fontsize="x-large", fontweight='bold')
    # plt.xlabel('Epoch',fontsize="x-large", fontweight='bold')
    # plt.legend(['DecFormer Train Loss', 'DecFormer Valid Loss', 'EncFormer Train Loss', 'EncFormer Valid Loss'],fontsize="x-large")
    # plt.axis([0, 51, 0, 8.5])
    # plt.grid()
    # save_file = os.path.join(result_save_dir, 'plot_train_valid_epoch_loss_transformers.png')
    # plt.savefig(save_file)    
    # # plt.show()

    plt.figure(5)
    ax1 = plt.subplot(121)
    
    ax1.set_title('DecFormer',fontsize="x-large", fontweight='bold')
    ax1.set_xlim([0, 51])
    ax1.set_ylim([0, 8.5])
    ax1.plot(epochs_train_decoder[:n_epochs],avg_train_loss_decoder[:n_epochs], epochs_val_decoder[:n_epochs], avg_valid_loss_decoder[:n_epochs])
    ax1.legend(['Train Loss', 'Valid Loss'])#,fontsize="large")
    ax1.set_ylabel('Loss',fontsize="large", fontweight='bold')
    ax1.set_xlabel('Epoch',fontsize="large", fontweight='bold')
    ax1.grid()
    
    ax2 = plt.subplot(122)
    ax2.set_title('EncFormer',fontsize="x-large", fontweight='bold')
    ax1.set_xlim([0, 51])
    ax1.set_ylim([0, 8.5])
    # ax2.set_axis([0, 51, 0, 8.5])
    ax2.plot(epochs_train_encoder[:n_epochs], avg_train_loss_encoder[:n_epochs], epochs_val_encoder[:n_epochs],avg_valid_loss_encoder[:n_epochs])
    ax2.legend(['Train Loss', 'Valid Loss'])#,fontsize="large")
    
    # plt.ylabel('Loss',fontsize="large", fontweight='bold')
    ax2.set_xlabel('Epoch',fontsize="large", fontweight='bold')    
    ax2.grid()

    save_file = os.path.join(result_save_dir, 'plot_train_valid_epoch_loss_transformers_v2.png')
    plt.savefig(save_file)    
    
    plt.show()

    



    train_acc_tags_decoder = all_train_log_tags_epoch_decoder[1:]
    valid_acc_tags_decoder = all_train_log_tags_valid_epoch_decoder[1:]

    train_acc_tags_encoder = all_train_log_tags_epoch_encoder[1:]
    valid_acc_tags_encoder = all_train_log_tags_valid_epoch_encoder[1:]

    # fig, axs = plt.subplots(2, 1, layout='constrained')
    

    plt.figure(2)
    plt.axis([0, 65, 0, 1.1])
    # fig, axs = plt.subplots()

    plt.subplot(211)
    # fig, axs = plt.subplot(2,1,2)
    plt.axis([0, 65, 0, 1.1])
    for train_acc_tag in train_acc_tags_decoder:
        filename_train = filename_prefix_decoder + train_acc_tag + filename_postfix
        epochs, avg_train_acc_top_k = read_csv_training_data(log_dir=logs_dir_decoder, filename=filename_train)
        plt.plot(epochs[:n_epochs], avg_train_acc_top_k[:n_epochs])

    for valid_acc_tag in valid_acc_tags_decoder:
        filename_train = filename_prefix_decoder + valid_acc_tag + filename_postfix
        epochs, avg_train_acc_top_k = read_csv_training_data(log_dir=logs_dir_decoder, filename=filename_train)
        plt.plot(epochs[:n_epochs], avg_train_acc_top_k[:n_epochs])

    # plt.legend(['Dec train top1', 'Dec train top3', 'Dec train top5', 'Dec val top1', 'Dec val top3', 'Dec val top5'])#,fontsize="large")
    plt.legend(['Train top1', 'Train top3', 'Train top5', 'Valid top1', 'Valid top3', 'Valid top5'])#,fontsize="large")
    plt.title('Decoder-Only Chess Transformer (DecFormer)',fontsize="x-large", fontweight='bold')
    plt.ylabel('Accuracy',fontsize="large", fontweight='bold')
    plt.grid()

    # plt.xlabel('Epoch',fontsize="x-large", fontweight='bold')

    # axs[0,0].title('DecFormer')
    # axs[0,0].set(xlabel='Epoch', ylabel='Accuracy',fontsize="x-large", fontweight='bold')


    plt.subplot(212)
    plt.axis([0, 65, 0, 1.1])
    for train_acc_tag in train_acc_tags_encoder:
        filename_train = filename_prefix_encoder + train_acc_tag + filename_postfix
        epochs, avg_train_acc_top_k = read_csv_training_data(log_dir=logs_dir_encoder, filename=filename_train)
        plt.plot(epochs[:n_epochs], avg_train_acc_top_k[:n_epochs])

    for valid_acc_tag in valid_acc_tags_encoder:
        filename_train = filename_prefix_encoder + valid_acc_tag + filename_postfix
        epochs, avg_train_acc_top_k = read_csv_training_data(log_dir=logs_dir_encoder, filename=filename_train)
        plt.plot(epochs[:n_epochs], avg_train_acc_top_k[:n_epochs])

    # plt.legend(['Enc train top1', 'Enc train top3', 'Enc train top5', 'Enc val top1', 'Enc val top3', 'Enc val top5'])#,fontsize="large")
    plt.legend(['Train top1', 'Train top3', 'Train top5', 'Valid top1', 'Valid top3', 'Valid top5'])#,fontsize="large")
    plt.title('Encoder-Only Chess Transformer (EncFormer)',fontsize="x-large", fontweight='bold')
    plt.ylabel('Accuracy',fontsize="large", fontweight='bold')
    plt.xlabel('Epoch',fontsize="large", fontweight='bold')
    plt.grid()

    save_file = os.path.join(result_save_dir, 'plot_train_valid_epoch_accuracies_transformers.png')
    plt.savefig(save_file)
    # plt.show()
    
    
    # axs[0].plot(epochs[:n_epochs], s1, epochs[:n_epochs], s2)
    # axs[0].set_xlim(0, 2)
    # axs[0].set_xlabel('Time (s)')
    # axs[0].set_ylabel('s1 and s2')
    # axs[0].grid(True)

    # cxy, f = axs[1].cohere(s1, s2, NFFT=256, Fs=1. / dt)
    # axs[1].set_ylabel('Coherence')

    # plt.show()
    # plt.ylabel('Accuracy',fontsize="x-large", fontweight='bold')
    # plt.xlabel('Epoch',fontsize="x-large", fontweight='bold')

    # # for ax in axs.flat:
    # #     for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
    # #         item.set_fontsize(20)
    # #     # ax.set(xlabel='Files', ylabel='Ranks', )
    # #     # Show all ticks and label them with the respective list entries
    # #     ax.set_xticks(range(len(files)), labels=files,
    # #                 rotation=0, ha="center", rotation_mode="anchor")
    # #     ax.set_yticks(range(len(ranks)), labels=ranks)

    # # axs[0].set_title('DecFormer')
    # # axs[1].set_title('EncFormer')
    # # axs[0].set_legend('Dec train top1', 'Dec train top3', 'Dec train top5', 'Dec val top1', 'Dec val top3', 'Dec val top5')
    # # axs[1].set_legend('Enc train top1', 'Enc train top3', 'Enc train top5', 'v val top1', 'Enc val top3', 'Enc val top5')
    # # plt.title(['DecFormer', 'EncFormer'])

    # # plt.legend(['DecFormer Train Top1', 'DecFormer Valid Top1', 'DecFormer Train Top3', 'DecFormer Valid Top3', 'DecFormer Train Top5', 'DecFormer Valid Top5', 'EncFormer Train Top1', 'EncFormer Valid Top1', 'EncFormer Train Top3', 'EncFormer Valid Top3', 'EncFormer Train Top5', 'EncFormer Val Top5'])
    # # plt.legend(['DecFormer Train Acc. Top1', 'DecFormer Valid Acc. Top1', 'DecFormer Train Acc. Top3', 'DecFormer Valid Acc. Top3', 'DecFormer Train Acc. Top5', 'DecFormer Valid Acc. Top5', 'EncFormer Train Acc. Top1', 'EncFormer Valid Acc. Top1', 'EncFormer Train Acc. Top3', 'EncFormer Valid Acc. Top3', 'EncFormer Train Acc. Top5', 'EncFormer Valid Acc. Top5'])
    # # plt.legend(['DecFormer Train Top1', 'DecFormer Train Top3', 'DecFormer Train Top5', 'DecFormer Valid Top1', 'DecFormer Valid Top3', 'DecFormer Valid Top5', 'EncFormer Train Top1', 'EncFormer Train Top3', 'EncFormer Train Top5', 'EncFormer Valid Top1', 'EncFormer Valid Top3', 'EncFormer Val Top5'],fontsize="large")
    # plt.legend(['Dec train top1', 'Dec train top3', 'Dec train top5', 'Dec val top1', 'Dec val top3', 'Dec val top5', 'Enc train top1', 'Enc train top3', 'Enc train top5', 'v val top1', 'Enc val top3', 'Enc val top5']) #,fontsize="large")
    # plt.legend(['Enc train top1', 'Enc train top3', 'Enc train top5', 'Enc val top1', 'Enc val top3', 'Enc val top5'],fontsize="large")
    # save_file = os.path.join(result_save_dir, 'plot_train_valid_epoch_accuracies_transformers.png')
    # plt.savefig(save_file)
    # # plt.show()

    # accuracy_names = ['Top1', 'Top3', 'Top5']

    # # idx = 0
    # plt.figure(2)

    # accuracy_names = ['Top1', 'Top3', 'Top5']
    # for train_acc_tag in train_acc_tags:
    #     filename_train = filename_prefix + train_acc_tag + filename_postfix
    #     epochs_train, avg_train_acc_top_k = read_csv_training_data(log_dir=logs_dir, filename=filename_train)

    #     # filename_valid = filename_prefix + valid_acc_tag + filename_postfix
    #     # epochs_valid, avg_valid_acc_top_k = read_csv_training_data(log_dir=logs_dir, filename=filename_valid)
        
    #     # plt.plot(epochs_train[:n_epochs], avg_train_acc_top_k[:n_epochs], epochs_valid[:n_epochs], avg_valid_acc_top_k[:n_epochs])
    #     plt.plot(epochs_train[:n_epochs], avg_train_acc_top_k[:n_epochs], linewidth=3)

    # plt.axis([0, 51, 0, 1.1])
    # plt.ylabel('Accuracy',fontsize="x-large", fontweight='bold')
    # plt.xlabel('Epoch',fontsize="x-large", fontweight='bold')
    # # plt.legend(['Train Acc. ' + accuracy_names[idx], 'Valid Acc. '  + accuracy_names[idx]],fontsize="large")
    # # plt.legend(['Train ' + accuracy_names[idx], 'Valid '  + accuracy_names[idx]],fontsize="x-large")
    # plt.legend(accuracy_names, fontsize='x-large')
    # plt.grid()
    # save_file = os.path.join(logs_dir, 'plot_train_epoch_acc_' + accuracy_names[idx] + '.png')
    # plt.savefig(save_file)
    
    # # plt.show()
        

    # plt.figure(3)
    # for valid_acc_tag in valid_acc_tags:
    #     filename_valid = filename_prefix + valid_acc_tag + filename_postfix
    #     epochs_valid, avg_valid_acc_top_k = read_csv_training_data(log_dir=logs_dir, filename=filename_valid)
        
    #     # plt.plot(epochs_train[:n_epochs], avg_train_acc_top_k[:n_epochs], epochs_valid[:n_epochs], avg_valid_acc_top_k[:n_epochs])
    #     plt.plot(epochs_valid[:n_epochs], avg_valid_acc_top_k[:n_epochs], linewidth=3)
        
    # plt.axis([0, 51, 0, 1.1])
    # plt.ylabel('Accuracy',fontsize="x-large", fontweight='bold')
    # plt.xlabel('Epoch',fontsize="x-large", fontweight='bold')
    # # plt.legend(['Train Acc. ' + accuracy_names[idx], 'Valid Acc. '  + accuracy_names[idx]],fontsize="large")
    # plt.legend(accuracy_names,fontsize="x-large")
    # plt.grid()
    # save_file = os.path.join(logs_dir, 'plot_valid_epoch_acc_' + accuracy_names[idx] + '.png')
    # plt.savefig(save_file)
    # # plt.show()